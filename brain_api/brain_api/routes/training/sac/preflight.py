"""SAC full-training readiness endpoint."""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, datetime, timedelta

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from brain_api.core.lstm import load_prices_yfinance
from brain_api.core.model_buckets import ModelType, UnknownBucketError, get_bucket
from brain_api.core.portfolio_rl.data_loading import (
    align_signals_to_weekly,
    missing_weekly_news_coverage,
    news_backfill_bounds,
)
from brain_api.core.portfolio_rl.weekly_patchtst import ready_weekly_snapshots
from brain_api.core.sac.momentum_signals import MOM_12_1_CALENDAR_BUFFER_DAYS
from brain_api.core.sac.readiness import SACReadinessIssue, SACTrainingReadiness
from brain_api.core.sac.trade_clock import (
    build_sac_weekly_trade_clock,
    extract_session_open_prices,
)
from brain_api.news.errors import NewsCoverageMissing
from brain_api.news.sync import NewsSyncError, sync_news
from brain_api.storage.forecaster_snapshots import SnapshotLocalStorage
from brain_api.storage.policy import (
    StoragePolicyError,
    get_prior_metadata_for_bucket,
)

from ._shared import SACTrainRequest, sac_current_is_reusable, sac_us_allowed_universes
from .forecast_dependency import SAC_PATCHTST_RECIPE, resolve_sac_training_window

router = APIRouter()
logger = logging.getLogger(__name__)


class SACReadinessIssueResponse(BaseModel):
    """Exact preflight failure with source and optional symbol."""

    source: str
    detail: str
    symbol: str | None = None
    retryable: bool


class SACTrainingReadinessResponse(BaseModel):
    """Readiness contract consumed by the durable Temporal loop."""

    universe: str
    symbols: list[str]
    ready: bool
    missing: list[SACReadinessIssueResponse]
    errors: list[SACReadinessIssueResponse]
    news_backfill_start: str | None = None
    news_backfill_end: str | None = None


def _required_snapshot_cutoffs(start_date: date, end_date: date) -> list[date]:
    return [
        date(year - 1, 12, 31) for year in range(start_date.year, end_date.year + 1)
    ]


def _news_issues_from_missing_cells(
    missing_cells: list[tuple[str, datetime, datetime]],
) -> list[SACReadinessIssue]:
    ends_by_symbol: dict[str, list] = defaultdict(list)
    for symbol, _start, end in missing_cells:
        ends_by_symbol[symbol].append(end)
    issues: list[SACReadinessIssue] = []
    for symbol, ends in ends_by_symbol.items():
        latest = max(ends)
        issues.append(
            SACReadinessIssue(
                "news",
                f"missing news coverage for {len(ends)} week(s); "
                f"latest {latest.isoformat()}",
                symbol=symbol,
                retryable=True,
            )
        )
    return issues


def assess_sac_training_readiness(
    universe: str, *, force: bool = False
) -> SACTrainingReadiness:
    """Validate the same strict price and signal inputs consumed by training."""
    bucket = get_bucket(ModelType.SAC, universe)
    try:
        sync_news()
    except NewsSyncError as exc:
        return SACTrainingReadiness.from_issues(
            universe=universe,
            symbols=[],
            missing=[],
            errors=[SACReadinessIssue("news_sync", str(exc), retryable=True)],
        )
    symbols = bucket.symbols_resolver()
    try:
        start_date, end_date = resolve_sac_training_window()
    except (ValueError, StoragePolicyError) as exc:
        return SACTrainingReadiness.from_issues(
            universe=universe,
            symbols=symbols,
            missing=[],
            errors=[SACReadinessIssue("patchtst_snapshot", str(exc), retryable=False)],
        )
    if not force:
        prior_metadata = get_prior_metadata_for_bucket(bucket=bucket)
        if sac_current_is_reusable(
            prior_metadata,
            symbols,
            forecast_recipe=SAC_PATCHTST_RECIPE,
            start_date=start_date.isoformat(),
        ):
            return SACTrainingReadiness.from_issues(
                universe=universe,
                symbols=symbols,
                missing=[],
                errors=[],
            )
    missing: list[SACReadinessIssue] = []
    errors: list[SACReadinessIssue] = []
    trade_clock = build_sac_weekly_trade_clock(start_date, end_date)
    weekly_cutoffs = trade_clock.transition_actor_cutoffs
    if trade_clock.transition_count < 208:
        return SACTrainingReadiness.from_issues(
            universe=universe,
            symbols=symbols,
            missing=[
                SACReadinessIssue(
                    "patchtst_snapshot",
                    "SAC needs at least 208 weekly transitions",
                    retryable=False,
                )
            ],
            errors=[],
        )

    try:
        # Fetch extra calendar history before start_date so the earliest
        # weekly cutoff still has enough trading bars for momentum_12_1
        # (skip 21 + lookback 252 = 273 bars) -- mirrors the buffer used
        # by the actual /train/sac/full price fetch.
        price_start_date = start_date - timedelta(days=MOM_12_1_CALENDAR_BUFFER_DAYS)
        prices = load_prices_yfinance(symbols, price_start_date, end_date)
    except Exception as exc:
        prices = {}
        errors.append(SACReadinessIssue("prices", str(exc), retryable=True))

    logger.info(
        "[SAC preflight] prices loaded %s/%s",
        sum(1 for symbol in symbols if prices.get(symbol) is not None),
        len(symbols),
    )

    price_ready_symbols: list[str] = []
    for symbol in symbols:
        price_frame = prices.get(symbol)
        price_ready = price_frame is not None and not price_frame.empty
        if not price_ready:
            missing.append(
                SACReadinessIssue(
                    "prices",
                    f"Missing daily price history for {symbol}",
                    symbol=symbol,
                    retryable=True,
                )
            )
            continue
        try:
            extract_session_open_prices(
                price_frame,
                trade_clock.rebalance_sessions,
                symbol=symbol,
            )
        except ValueError as exc:
            missing.append(
                SACReadinessIssue(
                    "prices",
                    str(exc),
                    symbol=symbol,
                    retryable=True,
                )
            )
            continue
        price_ready_symbols.append(symbol)

    news_complete = False
    logger.info(
        "[SAC preflight] news coverage grid %s symbols x %s weeks",
        len(symbols),
        len(weekly_cutoffs),
    )
    try:
        missing_cells = missing_weekly_news_coverage(symbols, weekly_cutoffs)
    except Exception as exc:
        errors.append(SACReadinessIssue("news", str(exc), retryable=True))
    else:
        logger.info(
            "[SAC preflight] news coverage missing %s cells", len(missing_cells)
        )
        if missing_cells:
            missing.extend(_news_issues_from_missing_cells(missing_cells))
        else:
            news_complete = True

    if news_complete and price_ready_symbols:
        logger.info(
            "[SAC preflight] aligning signals for %s symbols",
            len(price_ready_symbols),
        )
        try:
            align_signals_to_weekly(
                {symbol: prices[symbol] for symbol in price_ready_symbols},
                price_ready_symbols,
                weekly_cutoffs=weekly_cutoffs,
            )
        except NewsCoverageMissing as exc:
            missing.append(SACReadinessIssue("news", str(exc), retryable=True))
        except ValueError as exc:
            missing.append(SACReadinessIssue("prices", str(exc), retryable=True))

    try:
        ready_weekly_snapshots(
            SnapshotLocalStorage("patchtst_halal_new"), start_date, end_date
        )
    except ValueError as exc:
        missing.append(
            SACReadinessIssue("patchtst_snapshot", str(exc), retryable=False)
        )
    except StoragePolicyError as exc:
        errors.append(SACReadinessIssue("patchtst_snapshot", str(exc), retryable=True))

    backfill_start, backfill_end = news_backfill_bounds(weekly_cutoffs)
    return SACTrainingReadiness.from_issues(
        universe=universe,
        symbols=symbols,
        missing=missing,
        errors=errors,
        news_backfill_start=backfill_start,
        news_backfill_end=backfill_end,
    )


@router.post("/sac/preflight", response_model=SACTrainingReadinessResponse)
def preflight_sac_training(
    request: SACTrainRequest = SACTrainRequest(),
) -> SACTrainingReadinessResponse:
    """Return every exact missing/error condition before durable training."""
    if request.universe not in sac_us_allowed_universes():
        raise HTTPException(
            status_code=422,
            detail=f"Unknown SAC universe {request.universe!r}",
        )
    try:
        readiness = assess_sac_training_readiness(request.universe, force=request.force)
    except UnknownBucketError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return SACTrainingReadinessResponse(
        universe=readiness.universe,
        symbols=list(readiness.symbols),
        ready=readiness.ready,
        missing=[
            SACReadinessIssueResponse(**issue.to_dict()) for issue in readiness.missing
        ],
        errors=[
            SACReadinessIssueResponse(**issue.to_dict()) for issue in readiness.errors
        ],
        news_backfill_start=readiness.news_backfill_start,
        news_backfill_end=readiness.news_backfill_end,
    )

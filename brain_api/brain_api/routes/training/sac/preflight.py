"""SAC full-training readiness endpoint."""

from __future__ import annotations

from datetime import date, timedelta

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from brain_api.core.config import resolve_training_window
from brain_api.core.lstm import load_prices_yfinance
from brain_api.core.model_buckets import ModelType, UnknownBucketError, get_bucket
from brain_api.core.portfolio_rl.data_loading import (
    align_signals_to_weekly,
    news_backfill_bounds,
    require_weekly_news_coverage,
)
from brain_api.core.sac.momentum_signals import MOM_12_1_CALENDAR_BUFFER_DAYS
from brain_api.core.sac.readiness import SACReadinessIssue, SACTrainingReadiness
from brain_api.core.sac.trade_clock import (
    build_sac_weekly_trade_clock,
    extract_session_open_prices,
)
from brain_api.news.errors import NewsCoverageMissing
from brain_api.storage.forecaster_snapshots import SnapshotLocalStorage
from brain_api.storage.policy import (
    StoragePolicyError,
    ensure_snapshot_for_bucket,
    get_prior_metadata_for_bucket,
)

from ._shared import SACTrainRequest, sac_current_is_reusable, sac_us_allowed_universes

router = APIRouter()


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


def assess_sac_training_readiness(
    universe: str, *, force: bool = False
) -> SACTrainingReadiness:
    """Validate the same strict price and signal inputs consumed by training."""
    bucket = get_bucket(ModelType.SAC, universe)
    symbols = bucket.symbols_resolver()
    if not force:
        prior_metadata = get_prior_metadata_for_bucket(bucket=bucket)
        if sac_current_is_reusable(prior_metadata, symbols):
            return SACTrainingReadiness.from_issues(
                universe=universe,
                symbols=symbols,
                missing=[],
                errors=[],
            )
    start_date, end_date = resolve_training_window()
    missing: list[SACReadinessIssue] = []
    errors: list[SACReadinessIssue] = []
    trade_clock = build_sac_weekly_trade_clock(start_date, end_date)
    weekly_cutoffs = trade_clock.transition_actor_cutoffs

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
        else:
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
                price_ready = False

        news_ok = False
        try:
            require_weekly_news_coverage([symbol], weekly_cutoffs)
            news_ok = True
        except NewsCoverageMissing as exc:
            missing.append(
                SACReadinessIssue("news", str(exc), symbol=symbol, retryable=True)
            )
        except Exception as exc:
            errors.append(
                SACReadinessIssue("news", str(exc), symbol=symbol, retryable=True)
            )

        if price_ready and news_ok:
            try:
                align_signals_to_weekly(
                    {symbol: price_frame},
                    [symbol],
                    weekly_cutoffs=weekly_cutoffs,
                )
            except NewsCoverageMissing as exc:
                missing.append(
                    SACReadinessIssue("news", str(exc), symbol=symbol, retryable=True)
                )
            except ValueError as exc:
                missing.append(
                    SACReadinessIssue("prices", str(exc), symbol=symbol, retryable=True)
                )

    for forecaster_type in ("patchtst",):
        storage = SnapshotLocalStorage(forecaster_type)
        for cutoff in _required_snapshot_cutoffs(start_date, end_date):
            try:
                available = ensure_snapshot_for_bucket(
                    snapshot_storage=storage, cutoff_date=cutoff
                )
                if not available:
                    missing.append(
                        SACReadinessIssue(
                            f"{forecaster_type}_snapshot",
                            f"Missing walk-forward snapshot for {cutoff.isoformat()}",
                            retryable=False,
                        )
                    )
            except StoragePolicyError as exc:
                errors.append(
                    SACReadinessIssue(
                        f"{forecaster_type}_snapshot",
                        str(exc),
                        retryable=True,
                    )
                )

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

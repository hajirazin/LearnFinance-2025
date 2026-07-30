"""SAC full-training readiness endpoint."""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from brain_api.core.config import resolve_training_window
from brain_api.core.fundamentals import load_historical_fundamentals_from_cache
from brain_api.core.model_buckets import ModelType, UnknownBucketError, get_bucket
from brain_api.core.portfolio_rl.data_loading import load_historical_news_sentiment
from brain_api.core.sac.readiness import SACReadinessIssue, SACTrainingReadiness
from brain_api.storage.forecaster_snapshots import SnapshotLocalStorage
from brain_api.storage.policy import (
    StoragePolicyError,
    ensure_snapshot_for_bucket,
    get_prior_metadata_for_bucket,
)

from ._shared import SACTrainRequest, sac_us_allowed_universes

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


def _required_snapshot_cutoffs(start_date: date, end_date: date) -> list[date]:
    return [
        date(year - 1, 12, 31) for year in range(start_date.year, end_date.year + 1)
    ]


def assess_sac_training_readiness(
    universe: str, *, force: bool = False
) -> SACTrainingReadiness:
    """Inspect local evidence and snapshot availability without training."""
    bucket = get_bucket(ModelType.SAC, universe)
    symbols = bucket.symbols_resolver()
    if not force:
        prior_metadata = get_prior_metadata_for_bucket(bucket=bucket)
        if prior_metadata is not None and set(prior_metadata.get("symbols", [])) == set(
            symbols
        ):
            return SACTrainingReadiness.from_issues(
                universe=universe,
                symbols=symbols,
                missing=[],
                errors=[],
            )
    start_date, end_date = resolve_training_window()
    missing: list[SACReadinessIssue] = []
    errors: list[SACReadinessIssue] = []

    for symbol in symbols:
        try:
            load_historical_news_sentiment(
                [symbol], start_date=start_date, end_date=end_date
            )
        except FileNotFoundError as exc:
            missing.append(
                SACReadinessIssue("news", str(exc), symbol=symbol, retryable=True)
            )
        except Exception as exc:
            errors.append(
                SACReadinessIssue("news", str(exc), symbol=symbol, retryable=True)
            )

        try:
            fundamentals = load_historical_fundamentals_from_cache(
                [symbol], start_date=date.min, end_date=end_date
            )
            if symbol not in fundamentals or fundamentals[symbol].empty:
                missing.append(
                    SACReadinessIssue(
                        "fundamentals",
                        "No complete SEC-filing-date-enriched periods available",
                        symbol=symbol,
                        retryable=True,
                    )
                )
        except Exception as exc:
            errors.append(
                SACReadinessIssue(
                    "fundamentals", str(exc), symbol=symbol, retryable=True
                )
            )

    for forecaster_type in ("lstm", "patchtst"):
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

    return SACTrainingReadiness.from_issues(
        universe=universe,
        symbols=symbols,
        missing=missing,
        errors=errors,
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
    )

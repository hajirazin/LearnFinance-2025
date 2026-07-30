"""Training-data refresh API for provider-checked SAC inputs."""

import logging
from datetime import date

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from brain_api.core.config import DEFAULT_LOOKBACK_YEARS
from brain_api.core.data_freshness import ensure_fresh_training_data
from brain_api.etl.universe_registry import (
    UnknownETLUniverseError,
    get_etl_symbols,
)

router = APIRouter()
logger = logging.getLogger(__name__)


class RefreshTrainingDataRequest(BaseModel):
    """Request model for refreshing training data."""

    universe: str = Field(
        ...,
        description=(
            "Registered ETL universe string. Determines the symbol slate for "
            "sentiment gap fill and fundamentals refresh."
        ),
        examples=["halal_filtered"],
    )
    start_date: str | None = Field(
        None,
        description=(
            "Training window start date (YYYY-MM-DD). Defaults to January 1 "
            f"{DEFAULT_LOOKBACK_YEARS} years before the end date."
        ),
        examples=["2016-01-01"],
    )
    end_date: str | None = Field(
        None,
        description="Training window end date (YYYY-MM-DD). Defaults to today.",
        examples=["2026-01-31"],
    )


class RefreshTrainingDataResponse(BaseModel):
    """Response model for a successful training-data refresh."""

    sentiment_gaps_filled: int
    sentiment_gaps_remaining: int
    fundamentals_refreshed: list[str]
    fundamentals_skipped: list[str]
    fundamentals_failed: list[str]
    fundamentals_errors: dict[str, str] = Field(default_factory=dict)
    duration_seconds: float


def _parse_optional_date(value: str | None, *, field_name: str) -> date | None:
    """Parse one optional ISO date or raise the route's HTTP 400 contract."""
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {field_name} format: {exc}. Use YYYY-MM-DD.",
        ) from exc


@router.post("/refresh-training-data", response_model=RefreshTrainingDataResponse)
def refresh_training_data(
    request: RefreshTrainingDataRequest,
) -> RefreshTrainingDataResponse:
    """Refresh news and exact-filing-date fundamentals for one universe."""
    try:
        symbols = get_etl_symbols(request.universe)
    except UnknownETLUniverseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    logger.info(
        "[ETL Refresh] Resolved %d symbols from universe=%r",
        len(symbols),
        request.universe,
    )

    end_date = _parse_optional_date(request.end_date, field_name="end_date")
    if end_date is None:
        end_date = date.today()
    start_date = _parse_optional_date(request.start_date, field_name="start_date")
    if start_date is None:
        start_date = date(end_date.year - DEFAULT_LOOKBACK_YEARS, 1, 1)
    if start_date > end_date:
        raise HTTPException(
            status_code=400,
            detail="start_date must be before or equal to end_date",
        )

    result = ensure_fresh_training_data(
        universe=request.universe,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
    )
    if result.fundamentals_failed:
        raise HTTPException(
            status_code=503,
            detail={
                "source": "fundamentals",
                "failed_symbols": result.fundamentals_failed,
                "errors": result.fundamentals_errors,
                "message": (
                    "Fundamentals refresh did not produce usable enriched data"
                ),
            },
        )

    return RefreshTrainingDataResponse(
        sentiment_gaps_filled=result.sentiment_gaps_filled,
        sentiment_gaps_remaining=result.sentiment_gaps_remaining,
        fundamentals_refreshed=result.fundamentals_refreshed,
        fundamentals_skipped=result.fundamentals_skipped_today,
        fundamentals_failed=result.fundamentals_failed,
        fundamentals_errors=result.fundamentals_errors,
        duration_seconds=result.duration_seconds,
    )

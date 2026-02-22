"""Models for ETL endpoints."""

from pydantic import BaseModel, Field


class RefreshTrainingDataRequest(BaseModel):
    """Request for POST /etl/refresh-training-data endpoint."""

    start_date: str | None = Field(
        None, description="Training window start (YYYY-MM-DD), defaults to 10 years ago"
    )
    end_date: str | None = Field(
        None, description="Training window end (YYYY-MM-DD), defaults to today"
    )


class RefreshTrainingDataResponse(BaseModel):
    """Response from POST /etl/refresh-training-data endpoint."""

    sentiment_gaps_filled: int
    sentiment_gaps_remaining: int
    fundamentals_refreshed: list[str]
    fundamentals_skipped: list[str]
    fundamentals_failed: list[str]
    duration_seconds: float

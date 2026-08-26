"""HTTP models for the news domain."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class NewsWindowRequest(BaseModel):
    symbols: Annotated[list[str], Field(min_length=1)]
    start_exclusive: datetime
    end_inclusive: datetime


class NewsCoverageItem(BaseModel):
    symbol: str
    status: Literal["complete", "verified_empty"]
    event_count: int
    future_revision_excluded_count: int
    sentiment_model_revision: str


class NewsEventDTO(BaseModel):
    provider: str
    provider_article_id: str
    symbol: str
    created_at: datetime
    updated_at: datetime
    source: str
    sentiment_score: float
    p_positive: float
    p_negative: float
    p_neutral: float
    confidence: float
    scored_text_sha256: str
    sentiment_model: str
    sentiment_model_revision: str
    schema_version: int


class NewsWindowResult(BaseModel):
    start_exclusive: datetime
    end_inclusive: datetime
    coverage: list[NewsCoverageItem]
    events: list[NewsEventDTO]


class NewsBackfillRequest(BaseModel):
    symbols: Annotated[list[str], Field(min_length=1)]
    start: datetime
    end: datetime


class NewsBackfillAccepted(BaseModel):
    job_id: str
    status: str

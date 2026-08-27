"""Parse-only news DTOs. Temporal must not aggregate or score events."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

# Must match brain_api.news.models.NEWS_ARCHIVE_START_ISO.
NEWS_ARCHIVE_START_ISO = "2015-01-05T09:00:00-05:00"


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


class MondayDecisionWindowResponse(BaseModel):
    cutoff: datetime
    start_exclusive: datetime
    end_inclusive: datetime


class SACNewsSymbolAudit(BaseModel):
    symbol: str
    sentiment_score: float
    article_count: int
    coverage_status: Literal["complete", "verified_empty"]


class SACNewsAudit(BaseModel):
    as_of: datetime
    start_exclusive: datetime
    end_inclusive: datetime
    per_symbol: list[SACNewsSymbolAudit]


class NewsBackfillResponse(BaseModel):
    job_id: str
    status: str
    windows_done: int = 0
    windows_total: int = 0
    events_scored: int = 0
    error: str | None = None

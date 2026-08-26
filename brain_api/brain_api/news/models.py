"""News-domain value objects. No RL imports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

NEWS_SCHEMA_VERSION = 1
NEWS_PROVIDER = "alpaca_benzinga"
NEWS_SENTIMENT_MODEL = "ProsusAI/finbert"
NEWS_SENTIMENT_REVISION = "4556d13015211d73dccd3fdd39d39232506f3e43"
SCORING_SCHEMA_VERSION = 1
FINBERT_BATCH_SIZE = 32
FINBERT_MAX_LENGTH = 512
MAX_ARTICLES_PER_SYMBOL_WINDOW = 500
MAX_ARTICLES_PER_REQUEST = 10_000
CONFIDENCE_RECENCY_TAU_HOURS = 168.0
# First Monday 09:00 NY window Alpaca/Benzinga can materialize. PPO training
# must not request coverage before this instant.
NEWS_ARCHIVE_START_ISO = "2020-10-05T09:00:00-04:00"

CoverageStatus = Literal["complete", "verified_empty"]
JobStatus = Literal["pending", "running", "complete", "failed"]


@dataclass(frozen=True)
class NewsWindow:
    """Half-open created_at membership ``(start_exclusive, end_inclusive]``."""

    start_exclusive: datetime
    end_inclusive: datetime

    def __post_init__(self) -> None:
        if self.start_exclusive.tzinfo is None or self.end_inclusive.tzinfo is None:
            raise ValueError("NewsWindow bounds must be timezone-aware")
        if self.start_exclusive >= self.end_inclusive:
            raise ValueError("NewsWindow start_exclusive must be before end_inclusive")


@dataclass(frozen=True)
class ProviderArticle:
    """Transient Alpaca article. Headline/summary are not persisted."""

    provider_article_id: str
    symbol: str
    created_at: datetime
    updated_at: datetime
    source: str
    headline: str
    summary: str


@dataclass(frozen=True)
class NewsEvent:
    """One scored article-symbol fact. No raw text."""

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
    ingested_at: datetime


@dataclass(frozen=True)
class NewsCoverage:
    """Proof that one (symbol, exact window, scorer identity) query finished."""

    provider: str
    symbol: str
    window_start_exclusive: datetime
    window_end_inclusive: datetime
    schema_version: int
    sentiment_model: str
    sentiment_model_revision: str
    status: CoverageStatus
    page_count: int
    event_count: int
    future_revision_excluded_count: int
    fetched_at: datetime
    request_manifest_hash: str


@dataclass(frozen=True)
class NewsJob:
    """Resumable ETL job row."""

    job_id: str
    requested_start: datetime
    requested_end: datetime
    symbols_hash: str
    schema_version: int
    sentiment_revision: str
    status: JobStatus
    last_completed_symbol: str | None
    last_completed_window_end: datetime | None
    windows_done: int
    windows_total: int
    events_scored: int
    error: str | None
    created_at: datetime
    updated_at: datetime

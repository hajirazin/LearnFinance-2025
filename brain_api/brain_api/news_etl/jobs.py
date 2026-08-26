"""News ETL job identity helpers."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from brain_api.core.weekly_decision import canonical_monday_windows_contained_in
from brain_api.news.hashing import news_job_id, symbols_hash
from brain_api.news.models import (
    NEWS_SCHEMA_VERSION,
    NEWS_SENTIMENT_REVISION,
    NewsJob,
)
from brain_api.news.store import NewsStore, utcnow

__all__ = ["get_or_create_job", "job_windows_total", "mark_job", "news_job_id"]


def job_windows_total(symbols: Sequence[str], *, start: datetime, end: datetime) -> int:
    windows = canonical_monday_windows_contained_in(start, end)
    return len(sorted(set(symbols))) * len(windows)


def get_or_create_job(
    store: NewsStore,
    *,
    start: datetime,
    end: datetime,
    symbols: Sequence[str],
    windows_total: int,
) -> NewsJob:
    job_id = news_job_id(start=start, end=end, symbols=symbols)
    existing = store.get_job(job_id)
    if existing is not None:
        return existing
    now = utcnow()
    job = NewsJob(
        job_id=job_id,
        requested_start=start,
        requested_end=end,
        symbols_hash=symbols_hash(symbols),
        schema_version=NEWS_SCHEMA_VERSION,
        sentiment_revision=NEWS_SENTIMENT_REVISION,
        status="pending",
        last_completed_symbol=None,
        last_completed_window_end=None,
        windows_done=0,
        windows_total=windows_total,
        events_scored=0,
        error=None,
        created_at=now,
        updated_at=now,
    )
    store.upsert_job(job)
    return job


def mark_job(store: NewsStore, job: NewsJob, **updates: object) -> NewsJob:
    payload = {**job.__dict__, **updates, "updated_at": utcnow()}
    updated = NewsJob(**payload)  # type: ignore[arg-type]
    store.upsert_job(updated)
    return updated

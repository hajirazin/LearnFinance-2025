"""Serialize news ETL job rows for HTTP pollers."""

from __future__ import annotations

from brain_api.news.models import NewsJob


def job_to_dict(job: NewsJob) -> dict[str, object]:
    return {
        "job_id": job.job_id,
        "status": job.status,
        "requested_start": job.requested_start.isoformat(),
        "requested_end": job.requested_end.isoformat(),
        "symbols_hash": job.symbols_hash,
        "schema_version": job.schema_version,
        "sentiment_revision": job.sentiment_revision,
        "last_completed_symbol": job.last_completed_symbol,
        "last_completed_window_end": (
            job.last_completed_window_end.isoformat()
            if job.last_completed_window_end is not None
            else None
        ),
        "windows_done": job.windows_done,
        "windows_total": job.windows_total,
        "events_scored": job.events_scored,
        "error": job.error,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
    }

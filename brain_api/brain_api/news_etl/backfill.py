"""Canonical Monday-window backfill into the local DuckDB news store."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime

from brain_api.core.weekly_decision import canonical_monday_windows_contained_in
from brain_api.news.models import NewsWindow
from brain_api.news.service import NewsService
from brain_api.news.store import NewsStore
from brain_api.news_etl.jobs import get_or_create_job, job_windows_total, mark_job

logger = logging.getLogger(__name__)


def run_backfill(
    *,
    symbols: Sequence[str],
    start: datetime,
    end: datetime,
    store: NewsStore,
    service: NewsService | None = None,
) -> str:
    """Materialize every fully contained Monday window. Returns ``job_id``."""
    service = service or NewsService(store)
    ordered_symbols = sorted(set(symbols))
    windows = [
        NewsWindow(start_exclusive=start_exclusive, end_inclusive=end_inclusive)
        for start_exclusive, end_inclusive in canonical_monday_windows_contained_in(
            start, end
        )
    ]
    windows_total = job_windows_total(ordered_symbols, start=start, end=end)
    job = get_or_create_job(
        store,
        start=start,
        end=end,
        symbols=ordered_symbols,
        windows_total=windows_total,
    )
    if job.status == "complete":
        logger.info("news backfill already complete job_id=%s", job.job_id)
        return job.job_id
    logger.info(
        "news backfill start job_id=%s symbol_count=%s windows_per_symbol=%s coalesce=miss",
        job.job_id,
        len(ordered_symbols),
        len(windows),
    )
    job = mark_job(store, job, status="running", error=None)
    done = 0
    events_scored = 0
    try:
        for symbol in ordered_symbols:
            for window in windows:
                existing = store.get_coverage(symbol, window)
                if existing is not None:
                    done += 1
                    job = mark_job(
                        store,
                        job,
                        last_completed_symbol=symbol,
                        last_completed_window_end=window.end_inclusive,
                        windows_done=done,
                    )
                    continue
                coverage, events = service.materialize([symbol], window)
                done += 1
                events_scored += len(events)
                job = mark_job(
                    store,
                    job,
                    last_completed_symbol=symbol,
                    last_completed_window_end=window.end_inclusive,
                    windows_done=done,
                    events_scored=events_scored,
                )
                logger.info(
                    "news backfill symbol=%s window_end=%s status=%s events=%s done=%s/%s",
                    symbol,
                    window.end_inclusive.isoformat(),
                    coverage[0].status if coverage else "unknown",
                    len(events),
                    done,
                    windows_total,
                )
        mark_job(store, job, status="complete")
        logger.info(
            "news backfill end job_id=%s events_scored=%s", job.job_id, events_scored
        )
    except Exception as exc:
        logger.error("news backfill failed job_id=%s err=%s", job.job_id, exc)
        mark_job(store, job, status="failed", error=str(exc))
        raise
    return job.job_id

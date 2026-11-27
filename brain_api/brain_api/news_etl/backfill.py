"""Canonical Monday-window backfill into the local DuckDB news store."""

from __future__ import annotations

import fcntl
import logging
from collections.abc import Sequence
from datetime import datetime

from brain_api.news.models import NewsWindow
from brain_api.news.service import NewsService
from brain_api.news.store import NewsStore, coverage_key
from brain_api.news.sync import sync_news
from brain_api.news_etl.jobs import (
    get_or_create_job,
    job_windows_total,
    mark_job,
    news_job_id,
)
from brain_api.news_etl.windows import (
    NewsPurpose,
    missing_session_windows,
    requested_windows,
)

logger = logging.getLogger(__name__)


def run_backfill(
    *,
    symbols: Sequence[str],
    start: datetime,
    end: datetime,
    store: NewsStore,
    service: NewsService | None = None,
    shared: bool = False,
    purpose: NewsPurpose = "monday",
) -> str:
    """Run one worker per job, including across API requests and processes."""
    job_id = news_job_id(start=start, end=end, symbols=symbols, purpose=purpose)
    lock_path = store.db_path.parent / f"backfill-{job_id}.lock"
    with lock_path.open("a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logger.info(
                "news backfill already running job_id=%s; duplicate skipped", job_id
            )
            return job_id
        try:
            return _run_backfill(
                symbols=symbols,
                start=start,
                end=end,
                store=store,
                service=service,
                shared=shared,
                purpose=purpose,
            )
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _run_backfill(
    *,
    symbols: Sequence[str],
    start: datetime,
    end: datetime,
    store: NewsStore,
    service: NewsService | None,
    shared: bool,
    purpose: NewsPurpose,
) -> str:
    """Materialize uncovered Monday or PatchTST session windows."""
    service = service or NewsService(store)
    ordered_symbols = sorted(set(symbols))
    windows = requested_windows(start, end, purpose)
    windows_total = job_windows_total(
        ordered_symbols, start=start, end=end, purpose=purpose
    )
    job = get_or_create_job(
        store,
        start=start,
        end=end,
        symbols=ordered_symbols,
        windows_total=windows_total,
        purpose=purpose,
    )
    if shared:
        try:
            sync_news(store)
        except Exception as exc:
            mark_job(store, job, status="failed", error=str(exc))
            raise
    if job.status == "complete" and purpose == "monday":
        if shared:
            try:
                sync_news(store, publish=True)
            except Exception as exc:
                mark_job(store, job, status="failed", error=str(exc))
                raise
        logger.info("news backfill already complete job_id=%s", job.job_id)
        return job.job_id
    logger.info(
        "news backfill start job_id=%s symbol_count=%s windows_per_symbol=%s coalesce=miss",
        job.job_id,
        len(ordered_symbols),
        len(windows),
    )
    job = mark_job(store, job, status="running", error=None)
    existing = store.coverage_keys(ordered_symbols)
    session_gaps = (
        set(missing_session_windows(store, ordered_symbols, windows))
        if purpose == "patchtst"
        else set()
    )
    pending_by_window: list[tuple[NewsWindow, list[str]]] = []
    for window in windows:
        pending = [
            symbol
            for symbol in ordered_symbols
            if (
                (symbol, window) in session_gaps
                if purpose == "patchtst"
                else coverage_key(symbol, window.start_exclusive, window.end_inclusive)
                not in existing
            )
        ]
        if pending:
            pending_by_window.append((window, pending))
    done = windows_total - sum(len(pending) for _, pending in pending_by_window)
    events_scored = 0
    try:
        job = mark_job(store, job, windows_done=done, events_scored=events_scored)
        logger.info(
            "news backfill skip-summary pending_windows=%s skipped_cells=%s done=%s/%s",
            len(pending_by_window),
            done,
            done,
            windows_total,
        )
        if not pending_by_window:
            if shared:
                sync_news(store, publish=True)
            mark_job(store, job, status="complete")
            logger.info(
                "news backfill end job_id=%s events_scored=%s",
                job.job_id,
                events_scored,
            )
            return job.job_id
        for window, pending in pending_by_window:
            _coverage, events = service.materialize(pending, window)
            done += len(pending)
            events_scored += len(events)
            job = mark_job(
                store,
                job,
                last_completed_symbol=pending[-1],
                last_completed_window_end=window.end_inclusive,
                windows_done=done,
                events_scored=events_scored,
            )
            logger.info(
                "news backfill window_end=%s pending=%s events=%s done=%s/%s",
                window.end_inclusive.isoformat(),
                len(pending),
                len(events),
                done,
                windows_total,
            )
        if shared:
            sync_news(store, publish=True)
        if purpose == "patchtst":
            unresolved = missing_session_windows(store, ordered_symbols, windows)
            if unresolved:
                examples = ", ".join(
                    f"{symbol}@{window.end_inclusive.isoformat()}"
                    for symbol, window in unresolved[:5]
                )
                raise ValueError(
                    f"{len(unresolved)} PatchTST session news inputs remain unusable "
                    f"after fetching. Historical revisions may be unavailable. Examples: {examples}"
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

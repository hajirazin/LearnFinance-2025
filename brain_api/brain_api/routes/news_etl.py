"""New ETL HTTP: backfill and gaps into the DuckDB news store."""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from brain_api.news.store import NewsStore
from brain_api.news.sync import NewsSyncError, sync_news
from brain_api.news_etl.backfill import run_backfill
from brain_api.news_etl.gaps import missing_windows, run_gap_fill
from brain_api.news_etl.jobs import get_or_create_job, job_windows_total
from brain_api.news_etl.patchtst import refresh_patchtst_news
from brain_api.news_etl.progress import job_to_dict
from brain_api.news_etl.windows import NewsPurpose
from brain_api.storage.base import DEFAULT_DATA_PATH

from .training.job_registry import (
    complete_job,
    fail_job,
    get_job,
    update_progress,
)
from .training.job_registry import (
    get_or_create_job as get_operation_job,
)

logger = logging.getLogger(__name__)
router = APIRouter()


class NewsETLRequest(BaseModel):
    symbols: Annotated[list[str], Field(min_length=1)]
    start: datetime
    end: datetime
    purpose: NewsPurpose = "monday"


def get_news_store() -> NewsStore:
    return NewsStore(DEFAULT_DATA_PATH)


def _run_backfill(
    symbols: list[str], start: datetime, end: datetime, purpose: NewsPurpose = "monday"
) -> None:
    logger.info("news backfill worker start symbol_count=%s", len(symbols))
    try:
        run_backfill(
            symbols=symbols,
            start=start,
            end=end,
            store=get_news_store(),
            shared=True,
            purpose=purpose,
        )
    except Exception:
        logger.exception("news backfill background job failed")


def _run_gaps(
    symbols: list[str], start: datetime, end: datetime, purpose: NewsPurpose = "monday"
) -> None:
    try:
        run_gap_fill(
            symbols=symbols,
            start=start,
            end=end,
            store=get_news_store(),
            shared=True,
            purpose=purpose,
        )
    except Exception:
        logger.exception("news gap job failed")


@router.post("/news/backfill", status_code=202)
def start_news_backfill(
    request: NewsETLRequest, background_tasks: BackgroundTasks
) -> dict:
    store = get_news_store()
    job = get_or_create_job(
        store,
        start=request.start,
        end=request.end,
        symbols=request.symbols,
        purpose=request.purpose,
        windows_total=job_windows_total(
            request.symbols,
            start=request.start,
            end=request.end,
            purpose=request.purpose,
        ),
    )
    background_tasks.add_task(
        _run_backfill, request.symbols, request.start, request.end, request.purpose
    )
    logger.info(
        "news backfill queued job_id=%s status=%s symbol_count=%s",
        job.job_id,
        job.status,
        len(request.symbols),
    )
    return {"job_id": job.job_id, "status": job.status}


@router.get("/news/backfill/{job_id}")
def get_news_backfill(job_id: str) -> dict:
    job = get_news_store().get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown news job {job_id}")
    return job_to_dict(job)


@router.post("/news/gaps", status_code=202)
def start_news_gaps(request: NewsETLRequest, background_tasks: BackgroundTasks) -> dict:
    store = get_news_store()
    try:
        sync_news(store)
    except NewsSyncError as exc:
        raise HTTPException(503, str(exc)) from exc
    gaps = missing_windows(
        store,
        symbols=request.symbols,
        start=request.start,
        end=request.end,
        purpose=request.purpose,
    )
    logger.info("news gaps requested count=%s", len(gaps))
    job = get_or_create_job(
        store,
        start=request.start,
        end=request.end,
        symbols=request.symbols,
        purpose=request.purpose,
        windows_total=job_windows_total(
            request.symbols,
            start=request.start,
            end=request.end,
            purpose=request.purpose,
        ),
    )
    background_tasks.add_task(
        _run_gaps, request.symbols, request.start, request.end, request.purpose
    )
    return {
        "job_id": job.job_id,
        "status": job.status,
        "missing_windows": len(gaps),
        "purpose": request.purpose,
    }


@router.get("/news/gaps/{job_id}")
def get_news_gaps(job_id: str) -> dict:
    return get_news_backfill(job_id)


class NewsSyncRequest(BaseModel):
    publish: bool = False
    seed: bool = False


def _run_news_operation(job_id: str, operation, **kwargs) -> None:
    update_progress(job_id, {"phase": "syncing_news"})
    try:
        result = operation(get_news_store(), **kwargs)
    except Exception as exc:
        fail_job(job_id, str(exc))
        logger.exception("News operation %s failed", job_id)
    else:
        complete_job(job_id, {"news_commit": result})


def _queue_news_operation(background_tasks, kind, key, operation, **kwargs):
    job, created = get_operation_job(kind, key)
    if created:
        background_tasks.add_task(_run_news_operation, job.job_id, operation, **kwargs)
    return {"job_id": job.job_id, "status": job.status}


@router.post("/news/sync", status_code=202)
def start_news_sync(
    request: NewsSyncRequest, background_tasks: BackgroundTasks
) -> dict:
    action = "seed" if request.seed else "publish" if request.publish else "restore"
    return _queue_news_operation(
        background_tasks,
        "news_sync",
        action,
        sync_news,
        publish=request.publish,
        seed=request.seed,
    )


@router.get("/news/sync/{job_id}")
def get_news_sync(job_id: str) -> dict:
    job = get_job(job_id)
    if job is None or job.model_type not in ("news_sync", "patchtst_news"):
        raise HTTPException(status_code=404, detail=f"unknown news operation {job_id}")
    return {
        "job_id": job.job_id,
        "status": job.status,
        "result": job.result,
        "error": job.error,
    }


class PatchTSTNewsRequest(BaseModel):
    session_date: date


@router.post("/news/patchtst", status_code=202)
def start_patchtst_news(
    request: PatchTSTNewsRequest, background_tasks: BackgroundTasks
) -> dict:
    return _queue_news_operation(
        background_tasks,
        "patchtst_news",
        request.session_date.isoformat(),
        refresh_patchtst_news,
        session_date=request.session_date,
    )

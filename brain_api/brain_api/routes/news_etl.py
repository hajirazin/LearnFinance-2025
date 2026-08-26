"""New ETL HTTP: backfill and gaps into the DuckDB news store."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from brain_api.news.store import NewsStore
from brain_api.news_etl.backfill import run_backfill
from brain_api.news_etl.gaps import missing_windows, run_gap_fill
from brain_api.news_etl.jobs import get_or_create_job, job_windows_total
from brain_api.news_etl.progress import job_to_dict
from brain_api.storage.base import DEFAULT_DATA_PATH

logger = logging.getLogger(__name__)
router = APIRouter()


class NewsETLRequest(BaseModel):
    symbols: Annotated[list[str], Field(min_length=1)]
    start: datetime
    end: datetime


def get_news_store() -> NewsStore:
    return NewsStore(DEFAULT_DATA_PATH)


def _run_backfill(symbols: list[str], start: datetime, end: datetime) -> None:
    try:
        run_backfill(symbols=symbols, start=start, end=end, store=get_news_store())
    except Exception:
        logger.exception("news backfill background job failed")


def _run_gaps(symbols: list[str], start: datetime, end: datetime) -> None:
    try:
        run_gap_fill(symbols=symbols, start=start, end=end, store=get_news_store())
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
        windows_total=job_windows_total(
            request.symbols, start=request.start, end=request.end
        ),
    )
    background_tasks.add_task(
        _run_backfill, request.symbols, request.start, request.end
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
    gaps = missing_windows(
        store, symbols=request.symbols, start=request.start, end=request.end
    )
    logger.info("news gaps requested count=%s", len(gaps))
    job = get_or_create_job(
        store,
        start=request.start,
        end=request.end,
        symbols=request.symbols,
        windows_total=job_windows_total(
            request.symbols, start=request.start, end=request.end
        ),
    )
    background_tasks.add_task(_run_gaps, request.symbols, request.start, request.end)
    return {
        "job_id": job.job_id,
        "status": job.status,
        "missing_windows": len(gaps),
    }


@router.get("/news/gaps/{job_id}")
def get_news_gaps(job_id: str) -> dict:
    return get_news_backfill(job_id)

"""ETL endpoints for triggering batch pipelines.

Provides async job-based API for long-running ETL operations.
"""

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from brain_api.etl.config import ETLConfig
from brain_api.etl.pipeline import run_pipeline

router = APIRouter()


# ============================================================================
# Job State Management (in-memory for single-instance deployment)
# ============================================================================


@dataclass
class ETLJob:
    """Represents an ETL job and its state."""

    job_id: str
    status: Literal["pending", "running", "completed", "failed"]
    started_at: datetime
    completed_at: datetime | None = None
    progress: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    result: dict[str, Any] | None = None
    config: dict[str, Any] = field(default_factory=dict)


# Global job store (in-memory, suitable for single-instance)
_jobs: dict[str, ETLJob] = {}

# Maximum jobs to keep in memory
MAX_JOBS_IN_MEMORY = 100


def _cleanup_old_jobs() -> None:
    """Remove oldest jobs if we exceed the limit."""
    if len(_jobs) <= MAX_JOBS_IN_MEMORY:
        return

    # Sort by started_at, remove oldest
    sorted_jobs = sorted(_jobs.values(), key=lambda j: j.started_at)
    jobs_to_remove = len(_jobs) - MAX_JOBS_IN_MEMORY

    for job in sorted_jobs[:jobs_to_remove]:
        del _jobs[job.job_id]


def _update_job_progress(job_id: str, progress: dict[str, Any]) -> None:
    """Update job progress from the pipeline callback."""
    if job_id in _jobs:
        job = _jobs[job_id]
        job.progress = progress
        if progress.get("status") == "completed":
            job.status = "completed"
            job.completed_at = datetime.now(UTC)
            job.result = progress.get("output")


def _run_etl_job(job_id: str, config: ETLConfig) -> None:
    """Run the ETL pipeline in a background task."""
    job = _jobs.get(job_id)
    if not job:
        return

    job.status = "running"

    try:
        result = run_pipeline(
            config=config,
            progress_callback=lambda p: _update_job_progress(job_id, p),
        )
        job.status = "completed"
        job.completed_at = datetime.now(UTC)
        job.result = result
    except Exception as e:
        job.status = "failed"
        job.completed_at = datetime.now(UTC)
        job.error = str(e)


# ============================================================================
# Request / Response Models
# ============================================================================


class ETLJobRequest(BaseModel):
    """Request model for starting an ETL job."""

    batch_size: int = Field(
        256,
        ge=1,
        le=1024,
        description="Batch size for FinBERT processing (1-1024)",
    )
    max_articles: int | None = Field(
        None,
        ge=1,
        description="Maximum NEW articles to score (None = all)",
    )
    sentiment_threshold: float = Field(
        0.1,
        ge=0.0,
        le=1.0,
        description="Minimum |p_pos - p_neg| to include article (0.0-1.0)",
    )
    filter_to_halal: bool = Field(
        True,
        description="Filter to halal universe only",
    )
    local_only: bool = Field(
        True,
        description="Skip HuggingFace upload (local files only)",
    )
    output_dir: str = Field(
        "data/output",
        description="Output directory for parquet files",
    )
    cache_dir: str = Field(
        "data/cache",
        description="Directory for sentiment cache database",
    )


class ETLJobResponse(BaseModel):
    """Response model for job creation."""

    job_id: str
    status: str
    message: str


class ETLJobStatusResponse(BaseModel):
    """Response model for job status."""

    job_id: str
    status: str
    started_at: str
    completed_at: str | None
    progress: dict[str, Any]
    error: str | None
    result: dict[str, Any] | None
    config: dict[str, Any]


class ETLJobListResponse(BaseModel):
    """Response model for listing jobs."""

    jobs: list[ETLJobStatusResponse]
    total: int


# ============================================================================
# Endpoints
# ============================================================================


@router.post("/news-sentiment", response_model=ETLJobResponse, status_code=202)
def start_news_sentiment_etl(
    request: ETLJobRequest,
    background_tasks: BackgroundTasks,
) -> ETLJobResponse:
    """Start a news sentiment ETL job.

    This endpoint starts a long-running ETL pipeline that:
    1. Downloads the HuggingFace financial news dataset (if not cached)
    2. Filters to halal universe stocks
    3. Scores articles with FinBERT (with caching)
    4. Aggregates daily sentiment per symbol
    5. Outputs to parquet file

    The job runs asynchronously. Use GET /etl/news-sentiment/{job_id}
    to poll for status and results.

    Returns:
        ETLJobResponse with job_id for polling
    """
    # Clean up old jobs
    _cleanup_old_jobs()

    # Generate job ID
    job_id = str(uuid.uuid4())[:8]

    # Create config from request
    config = ETLConfig(
        batch_size=request.batch_size,
        max_articles=request.max_articles,
        sentiment_threshold=request.sentiment_threshold,
        filter_to_halal=request.filter_to_halal,
        local_only=request.local_only,
        output_dir=Path(request.output_dir),
        cache_dir=Path(request.cache_dir),
    )

    # Create job record
    job = ETLJob(
        job_id=job_id,
        status="pending",
        started_at=datetime.now(UTC),
        config={
            "batch_size": config.batch_size,
            "max_articles": config.max_articles,
            "sentiment_threshold": config.sentiment_threshold,
            "filter_to_halal": config.filter_to_halal,
            "local_only": config.local_only,
            "output_dir": str(config.output_dir),
            "cache_dir": str(config.cache_dir),
        },
    )
    _jobs[job_id] = job

    # Schedule background task
    background_tasks.add_task(_run_etl_job, job_id, config)

    return ETLJobResponse(
        job_id=job_id,
        status="pending",
        message=(
            f"ETL job {job_id} started. "
            f"Poll GET /etl/news-sentiment/{job_id} for status."
        ),
    )


@router.get("/news-sentiment/jobs", response_model=ETLJobListResponse)
def list_etl_jobs() -> ETLJobListResponse:
    """List all ETL jobs (most recent first).

    Returns:
        ETLJobListResponse with list of jobs
    """
    sorted_jobs = sorted(
        _jobs.values(),
        key=lambda j: j.started_at,
        reverse=True,
    )

    job_responses = [
        ETLJobStatusResponse(
            job_id=job.job_id,
            status=job.status,
            started_at=job.started_at.isoformat(),
            completed_at=job.completed_at.isoformat() if job.completed_at else None,
            progress=job.progress,
            error=job.error,
            result=job.result,
            config=job.config,
        )
        for job in sorted_jobs
    ]

    return ETLJobListResponse(
        jobs=job_responses,
        total=len(job_responses),
    )


@router.get("/news-sentiment/{job_id}", response_model=ETLJobStatusResponse)
def get_etl_job_status(job_id: str) -> ETLJobStatusResponse:
    """Get the status of an ETL job.

    Args:
        job_id: The job ID returned from POST /etl/news-sentiment

    Returns:
        ETLJobStatusResponse with current status and progress
    """
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return ETLJobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        started_at=job.started_at.isoformat(),
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        progress=job.progress,
        error=job.error,
        result=job.result,
        config=job.config,
    )



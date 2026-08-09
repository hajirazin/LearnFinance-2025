"""ETL endpoints for triggering batch pipelines.

Provides async job-based API for long-running ETL operations.
Background jobs respect the app-level shutdown_event so they stop
promptly when the server receives Ctrl+C / SIGINT.
"""

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from brain_api.core.config import DEFAULT_LOOKBACK_YEARS
from brain_api.etl.config import ETLConfig
from brain_api.etl.gap_fill import GapFillProgress, fill_sentiment_gaps
from brain_api.etl.pipeline import run_pipeline
from brain_api.etl.universe_registry import (
    UnknownETLUniverseError,
    list_universes,
)


def _get_shutdown_event() -> threading.Event:
    """Get the app-level shutdown event (late import to avoid circular deps)."""
    from brain_api.main import shutdown_event

    return shutdown_event


router = APIRouter()
logger = logging.getLogger(__name__)


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
            shutdown_event=_get_shutdown_event(),
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


def _validate_universe_or_422(universe: str) -> None:
    """Reject unknown ``universe`` values with HTTP 422.

    Raising ``HTTPException(422)`` here keeps the route handler's happy
    path small and matches the training endpoints' contract for
    unknown-universe inputs.
    """
    try:
        # Cheap registry lookup; we only care about validity here, not
        # the symbol list (resolvers may be expensive).
        if universe not in list_universes():
            raise UnknownETLUniverseError(
                f"No ETL universe registered for {universe!r}. "
                f"Valid universes: {sorted(list_universes())}"
            )
    except UnknownETLUniverseError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


class ETLJobRequest(BaseModel):
    """Request model for starting an ETL job."""

    universe: str = Field(
        ...,
        description=(
            "Registered ETL universe string (see ETL universe registry). "
            "Validated against the registry; unknown values return 422."
        ),
        examples=["halal_filtered"],
    )
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
    2. Filters to the universe specified in the request body
    3. Scores articles with FinBERT (with caching)
    4. Aggregates daily sentiment per symbol
    5. Outputs to parquet file

    The job runs asynchronously. Use GET /etl/news-sentiment/{job_id}
    to poll for status and results.

    Returns:
        ETLJobResponse with job_id for polling
    """
    _validate_universe_or_422(request.universe)
    # Clean up old jobs
    _cleanup_old_jobs()

    # Generate job ID
    job_id = str(uuid.uuid4())[:8]

    config = ETLConfig(
        universe=request.universe,
        batch_size=request.batch_size,
        max_articles=request.max_articles,
        sentiment_threshold=request.sentiment_threshold,
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
            "universe": config.universe,
            "batch_size": config.batch_size,
            "max_articles": config.max_articles,
            "sentiment_threshold": config.sentiment_threshold,
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


# ============================================================================
# Sentiment Gap Fill Endpoint
# ============================================================================


class SentimentGapsRequest(BaseModel):
    """Request model for sentiment gap fill."""

    universe: str = Field(
        ...,
        description=(
            "Registered ETL universe string (see ETL universe registry). "
            "Validated against the registry; unknown values return 422."
        ),
        examples=["halal_filtered"],
    )
    start_date: str | None = Field(
        None,
        description=(
            "Earliest date to check for gaps (YYYY-MM-DD). Defaults to January "
            f"1 {DEFAULT_LOOKBACK_YEARS} years before end_date."
        ),
        examples=["2011-01-01"],
    )
    end_date: str | None = Field(
        None,
        description="Latest date to check (YYYY-MM-DD, defaults to today)",
        examples=["2026-01-07"],
    )
    local_only: bool = Field(
        False,
        description="Skip HuggingFace upload (local files only)",
    )


def _update_gap_fill_progress(job_id: str, progress: GapFillProgress) -> None:
    """Update job progress from the gap fill callback."""
    if job_id in _jobs:
        job = _jobs[job_id]
        # Convert dataclass to dict for storage
        job.progress = {
            "total_gaps": progress.total_gaps,
            "gaps_fillable": progress.gaps_fillable,
            "gaps_pre_api_date": progress.gaps_pre_api_date,
            "api_calls_made": progress.api_calls_made,
            "articles_fetched": progress.articles_fetched,
            "articles_scored": progress.articles_scored,
            "rows_added": progress.rows_added,
            "remaining_gaps": progress.remaining_gaps,
            "checkpoints_saved": progress.checkpoints_saved,
            "status": progress.status,
            "current_phase": progress.current_phase,
            "error": progress.error,
        }
        # A progress callback is never authoritative for successful completion.
        # ``fill_sentiment_gaps`` still has to publish the parquet to Hugging Face
        # and return the final result.  Only ``_run_gap_fill_job`` may atomically
        # expose ``completed`` together with that result and its required hf_url.
        if progress.status == "cancelled":
            job.status = "failed"
            job.completed_at = datetime.now(UTC)
            job.error = "Cancelled by server shutdown"
        elif progress.status == "failed":
            job.status = "failed"
            job.completed_at = datetime.now(UTC)
            job.error = progress.error


def _run_gap_fill_job(
    job_id: str,
    universe: str,
    start_date: date,
    end_date: date,
    parquet_path: Path,
    local_only: bool = False,
) -> None:
    """Run the gap fill pipeline in a background task."""
    job = _jobs.get(job_id)
    if not job:
        return

    job.status = "running"
    started = time.monotonic()

    try:
        result = fill_sentiment_gaps(
            universe=universe,
            start_date=start_date,
            end_date=end_date,
            parquet_path=parquet_path,
            progress_callback=lambda p: _update_gap_fill_progress(job_id, p),
            local_only=local_only,
            shutdown_event=_get_shutdown_event(),
        )

        job.result = {
            "success": result.success,
            "parquet_updated": result.parquet_updated,
            "statistics": result.statistics,
            "hf_url": result.hf_url,
            "duration_seconds": time.monotonic() - started,
            "progress": {
                "total_gaps": result.progress.total_gaps,
                "gaps_fillable": result.progress.gaps_fillable,
                "gaps_pre_api_date": result.progress.gaps_pre_api_date,
                "api_calls_made": result.progress.api_calls_made,
                "articles_fetched": result.progress.articles_fetched,
                "articles_scored": result.progress.articles_scored,
                "rows_added": result.progress.rows_added,
                "remaining_gaps": result.progress.remaining_gaps,
                "checkpoints_saved": result.progress.checkpoints_saved,
            },
        }
        job.completed_at = datetime.now(UTC)
        if not result.success:
            job.error = result.progress.error
        # Publish terminal status last. Pollers that observe ``completed`` must
        # also observe the final result and its mandatory non-local ``hf_url``.
        job.status = "completed" if result.success else "failed"

    except Exception as e:
        job.status = "failed"
        job.completed_at = datetime.now(UTC)
        job.error = str(e)
        job.result = {"duration_seconds": time.monotonic() - started}


@router.post("/sentiment-gaps", response_model=ETLJobResponse, status_code=202)
def start_sentiment_gaps_fill(
    request: SentimentGapsRequest,
    background_tasks: BackgroundTasks,
) -> ETLJobResponse:
    """Start a sentiment gap fill job.

    This endpoint identifies missing sentiment data in the output parquet file
    and fills gaps by fetching news from Alpaca API and scoring with FinBERT.

    The job:
    1. Reads data/output/daily_sentiment.parquet
    2. Identifies missing (date, symbol) pairs for the universe in the request body
    3. Fetches news from Alpaca API (2015+ only, rate-limited to 200/min)
    4. Scores articles with FinBERT
    5. Appends new sentiment data to parquet

    Note: Gaps before 2015 cannot be filled (no free API has historical data).

    Returns:
        ETLJobResponse with job_id for polling
    """
    _validate_universe_or_422(request.universe)
    # Clean up old jobs
    _cleanup_old_jobs()

    # Parse dates. The default mirrors the configured SAC training window.
    if request.end_date:
        try:
            end_date = date.fromisoformat(request.end_date)
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid end_date format: {e}. Use YYYY-MM-DD.",
            ) from e
    else:
        end_date = date.today()

    if request.start_date:
        try:
            start_date = date.fromisoformat(request.start_date)
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid start_date format: {e}. Use YYYY-MM-DD.",
            ) from e
    else:
        start_date = date(end_date.year - DEFAULT_LOOKBACK_YEARS, 1, 1)

    if start_date > end_date:
        raise HTTPException(
            status_code=400,
            detail="start_date must be before or equal to end_date",
        )

    # Generate job ID
    job_id = str(uuid.uuid4())[:8]

    # Parquet path
    parquet_path = Path("data/output/daily_sentiment.parquet")

    # Create job record
    job = ETLJob(
        job_id=job_id,
        status="pending",
        started_at=datetime.now(UTC),
        config={
            "universe": request.universe,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "parquet_path": str(parquet_path),
            "local_only": request.local_only,
        },
    )
    _jobs[job_id] = job

    # Schedule background task
    background_tasks.add_task(
        _run_gap_fill_job,
        job_id,
        request.universe,
        start_date,
        end_date,
        parquet_path,
        request.local_only,
    )

    return ETLJobResponse(
        job_id=job_id,
        status="pending",
        message=(
            f"Sentiment gap fill job {job_id} started. "
            f"Poll GET /etl/sentiment-gaps/{job_id} for status."
        ),
    )


@router.get("/sentiment-gaps/{job_id}", response_model=ETLJobStatusResponse)
def get_sentiment_gaps_job_status(job_id: str) -> ETLJobStatusResponse:
    """Get the status of a sentiment gap fill job.

    Args:
        job_id: The job ID returned from POST /etl/sentiment-gaps

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

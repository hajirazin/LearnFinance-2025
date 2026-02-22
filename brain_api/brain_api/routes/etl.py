"""ETL endpoints for triggering batch pipelines.

Provides async job-based API for long-running ETL operations.
Background jobs respect the app-level shutdown_event so they stop
promptly when the server receives Ctrl+C / SIGINT.
"""

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from brain_api.etl.config import ETLConfig
from brain_api.etl.gap_fill import GapFillProgress, fill_sentiment_gaps
from brain_api.etl.pipeline import run_pipeline


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
    2. Filters to the configured universe (ETL_UNIVERSE env var)
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

    # Create config from request (universe comes from ETL_UNIVERSE env var)
    config = ETLConfig(
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
            "batch_size": config.batch_size,
            "max_articles": config.max_articles,
            "sentiment_threshold": config.sentiment_threshold,
            "universe": config.universe.value,
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

    start_date: str = Field(
        ...,
        description="Earliest date to check for gaps (YYYY-MM-DD)",
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
        if progress.status == "completed":
            job.status = "completed"
            job.completed_at = datetime.now(UTC)
        elif progress.status == "cancelled":
            job.status = "failed"
            job.completed_at = datetime.now(UTC)
            job.error = "Cancelled by server shutdown"
        elif progress.status == "failed":
            job.status = "failed"
            job.completed_at = datetime.now(UTC)
            job.error = progress.error


def _run_gap_fill_job(
    job_id: str,
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

    try:
        result = fill_sentiment_gaps(
            start_date=start_date,
            end_date=end_date,
            parquet_path=parquet_path,
            progress_callback=lambda p: _update_gap_fill_progress(job_id, p),
            local_only=local_only,
            shutdown_event=_get_shutdown_event(),
        )

        job.status = "completed" if result.success else "failed"
        job.completed_at = datetime.now(UTC)
        job.result = {
            "success": result.success,
            "parquet_updated": result.parquet_updated,
            "statistics": result.statistics,
            "hf_url": result.hf_url,
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
        if not result.success:
            job.error = result.progress.error

    except Exception as e:
        job.status = "failed"
        job.completed_at = datetime.now(UTC)
        job.error = str(e)


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
    2. Identifies missing (date, symbol) pairs for the configured universe (ETL_UNIVERSE env var)
    3. Fetches news from Alpaca API (2015+ only, rate-limited to 200/min)
    4. Scores articles with FinBERT
    5. Appends new sentiment data to parquet

    Note: Gaps before 2015 cannot be filled (no free API has historical data).

    Returns:
        ETLJobResponse with job_id for polling
    """
    # Clean up old jobs
    _cleanup_old_jobs()

    # Parse dates
    try:
        start_date = date.fromisoformat(request.start_date)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid start_date format: {e}. Use YYYY-MM-DD.",
        ) from e

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


# ============================================================================
# Training Data Refresh Endpoint
# ============================================================================


class RefreshTrainingDataRequest(BaseModel):
    """Request model for refreshing training data."""

    start_date: str | None = Field(
        None,
        description="Training window start date (YYYY-MM-DD). Defaults to Jan 1st, 10 years ago.",
        examples=["2016-01-01"],
    )
    end_date: str | None = Field(
        None,
        description="Training window end date (YYYY-MM-DD). Defaults to today.",
        examples=["2026-01-31"],
    )


class RefreshTrainingDataResponse(BaseModel):
    """Response model for training data refresh."""

    sentiment_gaps_filled: int = Field(
        description="Number of sentiment gaps filled (2015+ only)"
    )
    sentiment_gaps_remaining: int = Field(
        description="Gaps that couldn't be filled (pre-2015)"
    )
    fundamentals_refreshed: list[str] = Field(
        description="Symbols whose fundamentals were refreshed"
    )
    fundamentals_skipped: list[str] = Field(
        description="Symbols already fetched today (skipped)"
    )
    fundamentals_failed: list[str] = Field(description="Symbols that failed to refresh")
    duration_seconds: float = Field(
        description="Total time taken for refresh operation"
    )


@router.post("/refresh-training-data", response_model=RefreshTrainingDataResponse)
def refresh_training_data(
    request: RefreshTrainingDataRequest,
) -> RefreshTrainingDataResponse:
    """Refresh training data (sentiment gaps + fundamentals).

    Symbols are resolved from the ETL_UNIVERSE config, ensuring this endpoint
    is self-sufficient and does not require the caller to specify symbols.

    This endpoint ensures training data is fresh before training by:
    1. Filling news sentiment gaps (2015+ via Alpaca API)
    2. Refreshing fundamentals not fetched today (via Alpha Vantage API)

    Returns:
        RefreshTrainingDataResponse with statistics on what was refreshed
    """
    from brain_api.core.data_freshness import ensure_fresh_training_data
    from brain_api.routes.training.dependencies import get_etl_symbols

    symbols = get_etl_symbols()
    logger.info(
        f"[ETL Refresh] Resolved {len(symbols)} symbols from ETL_UNIVERSE config"
    )

    # Parse end_date (default: today)
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

    # Parse start_date (default: Jan 1st, 15 years ago)
    if request.start_date:
        try:
            start_date = date.fromisoformat(request.start_date)
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid start_date format: {e}. Use YYYY-MM-DD.",
            ) from e
    else:
        from brain_api.core.config import DEFAULT_LOOKBACK_YEARS

        start_date = date(end_date.year - DEFAULT_LOOKBACK_YEARS, 1, 1)

    if start_date > end_date:
        raise HTTPException(
            status_code=400,
            detail="start_date must be before or equal to end_date",
        )

    # Call the shared data freshness function
    result = ensure_fresh_training_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
    )

    return RefreshTrainingDataResponse(
        sentiment_gaps_filled=result.sentiment_gaps_filled,
        sentiment_gaps_remaining=result.sentiment_gaps_remaining,
        fundamentals_refreshed=result.fundamentals_refreshed,
        fundamentals_skipped=result.fundamentals_skipped_today,
        fundamentals_failed=result.fundamentals_failed,
        duration_seconds=result.duration_seconds,
    )

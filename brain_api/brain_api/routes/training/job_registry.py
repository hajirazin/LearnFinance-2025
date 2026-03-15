"""In-memory training job registry for async training endpoints.

Tracks background training jobs so callers can poll for status.
Same pattern as the ETL job registry in routes/etl.py.
"""

import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

_training_jobs: dict[str, "TrainingJob"] = {}
_lock = threading.Lock()
MAX_JOBS = 50


@dataclass
class TrainingJob:
    """Represents a background training job and its state."""

    job_id: str
    model_type: str
    status: Literal["pending", "running", "completed", "failed", "cancelled"]
    started_at: datetime
    completed_at: datetime | None = None
    progress: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    result: dict[str, Any] | None = None


def _cleanup_old_jobs() -> None:
    """Remove oldest completed/failed/cancelled jobs if we exceed the limit."""
    if len(_training_jobs) <= MAX_JOBS:
        return
    terminal = [
        j
        for j in _training_jobs.values()
        if j.status in ("completed", "failed", "cancelled")
    ]
    terminal.sort(key=lambda j: j.started_at)
    jobs_to_remove = len(_training_jobs) - MAX_JOBS
    for job in terminal[:jobs_to_remove]:
        del _training_jobs[job.job_id]


def get_or_create_job(model: str, version: str) -> tuple[TrainingJob, bool]:
    """Get existing job or create a new one.

    Returns:
        (job, is_new) -- if a running/pending job for this version exists,
        returns it with is_new=False. Failed/cancelled jobs are replaced.
    """
    job_id = f"{model}:{version}"
    with _lock:
        existing = _training_jobs.get(job_id)
        if existing and existing.status in ("pending", "running"):
            return existing, False

        _cleanup_old_jobs()
        job = TrainingJob(
            job_id=job_id,
            model_type=model,
            status="pending",
            started_at=datetime.now(UTC),
        )
        _training_jobs[job_id] = job
        return job, True


def get_job(job_id: str) -> TrainingJob | None:
    """Look up a job by ID."""
    return _training_jobs.get(job_id)


def update_progress(job_id: str, progress: dict[str, Any]) -> None:
    """Update job progress (called from background thread)."""
    with _lock:
        job = _training_jobs.get(job_id)
        if job and job.status in ("pending", "running"):
            job.status = "running"
            job.progress = progress


def complete_job(job_id: str, result: dict[str, Any]) -> None:
    """Mark job as completed with the training result."""
    with _lock:
        job = _training_jobs.get(job_id)
        if job:
            job.status = "completed"
            job.completed_at = datetime.now(UTC)
            job.result = result


def fail_job(job_id: str, error: str) -> None:
    """Mark job as failed."""
    with _lock:
        job = _training_jobs.get(job_id)
        if job:
            job.status = "failed"
            job.completed_at = datetime.now(UTC)
            job.error = error


def cancel_job(job_id: str) -> None:
    """Mark job as cancelled (e.g. by shutdown)."""
    with _lock:
        job = _training_jobs.get(job_id)
        if job and job.status in ("pending", "running"):
            job.status = "cancelled"
            job.completed_at = datetime.now(UTC)
            job.error = "Cancelled by server shutdown"


def cancel_all_running_jobs() -> None:
    """Cancel all running/pending jobs. Called during server shutdown."""
    with _lock:
        for job in _training_jobs.values():
            if job.status in ("pending", "running"):
                job.status = "cancelled"
                job.completed_at = datetime.now(UTC)
                job.error = "Cancelled by server shutdown"

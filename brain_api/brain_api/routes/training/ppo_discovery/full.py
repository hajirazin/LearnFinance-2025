"""``POST /train/ppo-discovery/full`` — candidate only, never auto-promotes."""

from __future__ import annotations

from datetime import UTC, date, datetime

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from brain_api.core.ppo_discovery.config import UNIVERSE_NAME, PPODiscoveryConfig
from brain_api.core.ppo_discovery.pipeline import run_ppo_discovery_training
from brain_api.core.ppo_discovery.schemas import PPODiscoveryError
from brain_api.core.ppo_discovery.splits import resolve_experiment_variant
from brain_api.core.ppo_discovery.universe_snapshot import (
    load_universe_snapshot,
    resolve_universe_snapshot,
)
from brain_api.routes.training.job_registry import (
    complete_job,
    fail_job,
    get_or_create_job,
    update_progress,
)
from brain_api.routes.training.models import TrainingJobResponse
from brain_api.storage.base import DEFAULT_DATA_PATH
from brain_api.storage.ppo_discovery.local import PPODiscoveryHalalNewModelStorage

router = APIRouter()


class PPOTrainRequest(BaseModel):
    universe: str = Field(...)
    end_date: str | None = None
    start_date: str | None = None
    experiment_id: str = "ppo-discovery-default"
    snapshot_sha256: str | None = None
    total_timesteps: int | None = None
    seeds: list[int] | None = None


def _load_training_snapshot(request: PPOTrainRequest):
    if request.snapshot_sha256:
        return load_universe_snapshot(request.snapshot_sha256)
    return resolve_universe_snapshot(datetime.now(UTC), persist=True)


def _run_training(job_id: str, request: PPOTrainRequest) -> None:
    try:
        update_progress(job_id, {"stage": "freeze_universe"})
        snapshot = _load_training_snapshot(request)
        config = PPODiscoveryConfig()
        if request.total_timesteps is not None:
            config.total_timesteps = request.total_timesteps
        if request.seeds is not None:
            config.seeds = tuple(request.seeds)
        end = (
            date.fromisoformat(request.end_date)
            if request.end_date
            else datetime.now(UTC).date()
        )
        start = date.fromisoformat(request.start_date) if request.start_date else None
        storage = PPODiscoveryHalalNewModelStorage(base_path=DEFAULT_DATA_PATH)
        result = run_ppo_discovery_training(
            snapshot,
            config=config,
            storage=storage,
            end_date=end,
            start_date=start,
            experiment_id=request.experiment_id,
            experiment_variant=resolve_experiment_variant(config),
            progress=lambda payload: update_progress(job_id, payload),
            base_path=DEFAULT_DATA_PATH,
        )
        complete_job(job_id, result)
    except PPODiscoveryError as exc:
        fail_job(job_id, str(exc))
    except Exception as exc:
        fail_job(job_id, str(exc))


@router.post("/ppo-discovery/full")
def train_ppo_discovery_endpoint(
    background_tasks: BackgroundTasks,
    request: PPOTrainRequest,
) -> JSONResponse:
    if request.universe != UNIVERSE_NAME:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown universe '{request.universe}' for ppo_discovery",
        )
    version_hint = request.experiment_id
    job, is_new = get_or_create_job("ppo_discovery", version_hint)
    if is_new:
        background_tasks.add_task(_run_training, job.job_id, request)
    return JSONResponse(
        status_code=202,
        content=TrainingJobResponse(
            job_id=job.job_id,
            status=job.status,
            message="candidate training started; promotion is manual",
        ).model_dump(),
    )

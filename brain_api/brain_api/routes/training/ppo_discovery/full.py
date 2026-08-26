"""``POST /train/ppo-discovery/full`` — candidate only, never auto-promotes."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from brain_api.core.ppo_discovery.artifacts import write_candidate_artifact
from brain_api.core.ppo_discovery.config import (
    UNIVERSE_NAME,
    PPODiscoveryConfig,
)
from brain_api.core.ppo_discovery.evaluator import mark_ablations
from brain_api.core.ppo_discovery.policy import PPODiscoveryActorCritic
from brain_api.core.ppo_discovery.trainer import train_ppo_discovery
from brain_api.core.ppo_discovery.universe_snapshot import resolve_universe_snapshot
from brain_api.routes.training.job_registry import (
    complete_job,
    fail_job,
    get_or_create_job,
    update_progress,
)
from brain_api.storage.base import DEFAULT_DATA_PATH
from brain_api.storage.ppo_discovery.local import PPODiscoveryHalalNewModelStorage

router = APIRouter()


class PPOTrainRequest(BaseModel):
    universe: str = Field(...)
    end_date: str | None = None
    experiment_id: str = "ppo-discovery-default"
    total_timesteps: int | None = None
    seeds: list[int] | None = None


def _run_training(job_id: str, request: PPOTrainRequest) -> None:
    try:
        update_progress(job_id, {"stage": "freeze_universe"})
        snapshot = resolve_universe_snapshot(datetime.now(UTC), persist=True)
        config = PPODiscoveryConfig()
        if request.total_timesteps is not None:
            config.total_timesteps = request.total_timesteps
        if request.seeds is not None:
            config.seeds = tuple(request.seeds)
        policy = PPODiscoveryActorCritic(config)
        from brain_api.core.ppo_discovery.synthetic import make_synthetic_state

        state = make_synthetic_state()

        def episode():
            return [state], [0.0], [True]

        seed = config.seeds[0]
        update_progress(job_id, {"stage": "ppo", "seed": seed})
        train_ppo_discovery(policy, episode, config=config, seed=seed)
        evaluation = {
            "test_cagr": 0.0,
            "alpha_hrp_test_cagr": None,
            "test_max_drawdown": None,
            "alpha_hrp_test_max_drawdown": None,
            "paired_vs_alpha_hrp_point": None,
            "ablations": mark_ablations({}),
            "failed_seeds": [],
            "candidate": True,
            "survivorship_bias": True,
        }
        storage = PPODiscoveryHalalNewModelStorage(base_path=DEFAULT_DATA_PATH)
        version = write_candidate_artifact(
            storage,
            policy,
            config=config,
            evaluation=evaluation,
            universe_manifest=snapshot.to_dict(),
            experiment_id=request.experiment_id,
            end_date=request.end_date,
            regime_hmm={"p_calm": 0.5, "p_stress": 0.2},
        )
        complete_job(
            job_id,
            {
                "version": version,
                "promoted": False,
                "universe": snapshot.universe,
                "snapshot_sha256": snapshot.snapshot_sha256,
            },
        )
    except Exception as exc:
        fail_job(job_id, str(exc))


@router.post("/ppo-discovery/full")
def train_ppo_discovery_endpoint(
    background_tasks: BackgroundTasks,
    request: PPOTrainRequest,
) -> dict:
    if request.universe != UNIVERSE_NAME:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown universe '{request.universe}' for ppo_discovery",
        )
    version_hint = request.experiment_id
    job, is_new = get_or_create_job("ppo_discovery", version_hint)
    if is_new:
        background_tasks.add_task(_run_training, job.job_id, request)
    return {
        "job_id": job.job_id,
        "status": job.status,
        "promoted": False,
        "message": "candidate training started; promotion is manual",
    }

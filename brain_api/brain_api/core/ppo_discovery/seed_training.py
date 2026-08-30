"""Per-seed PPO loop: ledger, resume, validation, median selection."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from brain_api.core.ppo_discovery.checkpoints import (
    bound_error_message,
    load_seed_checkpoint,
    load_seed_partial_checkpoint,
    save_seed_checkpoint,
    write_seed_metadata,
)
from brain_api.core.ppo_discovery.config import PPODiscoveryConfig
from brain_api.core.ppo_discovery.environment import (
    WeeklyTransition,
    collect_closed_loop_rollout,
)
from brain_api.core.ppo_discovery.evaluator import (
    aggregate_seed_metrics,
    evaluate_policy_weeks,
    select_candidate_seed,
)
from brain_api.core.ppo_discovery.policy import PPODiscoveryActorCritic
from brain_api.core.ppo_discovery.schemas import PPODiscoveryError, UniverseSnapshot
from brain_api.core.ppo_discovery.seed_ledger import (
    complete_seed_rows,
    fail_job_on_accelerator_oom,
    failed_seed_ids,
    load_seeds_ledger,
    record_episode_partial,
    upsert_seed_row,
    write_seeds_ledger,
)
from brain_api.core.ppo_discovery.trainer import train_ppo_discovery
from brain_api.core.training_utils import is_accelerator_out_of_memory

logger = logging.getLogger(__name__)


def _log(message: str) -> None:
    print(message, flush=True)
    logger.info(message)


@dataclass(frozen=True)
class PPOSeedTrainingResult:
    selected_seed: int
    selected_policy: PPODiscoveryActorCritic
    seed_metrics: dict[str, dict[str, float]]
    seed_aggregates: dict[str, Any]
    failed_seeds: list[int]
    ledger: dict[str, Any]


def week_logs(policy, weeks, snapshot, ohlcv, spy, scalers, config) -> list[float]:
    steps = collect_closed_loop_rollout(
        policy,
        weeks,
        snapshot=snapshot,
        ohlcv_by_symbol=ohlcv,
        spy=spy,
        feature_scalers=scalers,
        config=config,
        deterministic=True,
    )
    return [step.realized_net_return for step in steps]


def eval_weeks(
    policy, weeks, snapshot, ohlcv, spy, scalers, config
) -> dict[str, float]:
    return evaluate_policy_weeks(
        week_logs(policy, weeks, snapshot, ohlcv, spy, scalers, config)
    )


def train_ppo_discovery_seeds(
    *,
    pretrained_state: Mapping[str, torch.Tensor],
    train_weeks: Sequence[WeeklyTransition],
    val_weeks: Sequence[WeeklyTransition],
    snapshot: UniverseSnapshot,
    ohlcv: Mapping,
    spy,
    scalers: Mapping[str, Any],
    config: PPODiscoveryConfig,
    ckpt_dir: Path,
    checkpoint_expected: Mapping[str, Any],
    experiment_id: str,
    device: torch.device,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> PPOSeedTrainingResult:
    """Train declared seeds. Only complete seeds enter median selection."""
    report = progress or (lambda _payload: None)
    ledger = load_seeds_ledger(ckpt_dir)
    seed_policies: dict[int, PPODiscoveryActorCritic] = {}
    last_error: Exception | None = None

    def _episode(current: PPODiscoveryActorCritic, cache):
        return collect_closed_loop_rollout(
            current,
            train_weeks,
            snapshot=snapshot,
            ohlcv_by_symbol=ohlcv,
            spy=spy,
            feature_scalers=scalers,
            config=config,
            temporal_cache=cache,
        )

    for seed in config.seeds:
        seed_i = int(seed)
        report(
            {
                "stage": "ppo",
                "seed": seed_i,
                "seed_status": "in_progress",
                "device": device.type,
                "completed_seeds": sorted(complete_seed_rows(ledger)),
                "failed_seeds": failed_seed_ids(ledger),
            }
        )
        ledger = upsert_seed_row(
            ledger, seed_i, status="in_progress", device=device.type
        )
        write_seeds_ledger(ckpt_dir, ledger)
        seed_policy = PPODiscoveryActorCritic(config).to(device)
        seed_policy.load_state_dict(pretrained_state)
        seed_policy.to(device)
        counters = {"steps_done": 0, "episode_index": 0, "update_index": 0}
        try:
            loaded = load_seed_checkpoint(
                ckpt_dir, seed=seed_i, expected=dict(checkpoint_expected)
            )
            loaded_status = None if loaded is None else loaded["metadata"].get("status")
            if loaded is not None and loaded_status == "complete":
                seed_policy.load_state_dict(loaded["state_dict"])
                seed_policy.to(device)
                report(
                    {
                        "stage": "ppo_resume",
                        "seed": seed_i,
                        "seed_status": "complete",
                    }
                )
                ledger = upsert_seed_row(
                    ledger,
                    seed_i,
                    status="complete",
                    val_cagr=loaded["metadata"].get("val_cagr"),
                    val_sharpe=loaded["metadata"].get("val_sharpe"),
                    device=device.type,
                )
                write_seeds_ledger(ckpt_dir, ledger)
            else:
                if loaded is not None:
                    seed_policy.load_state_dict(loaded["state_dict"])
                    seed_policy.to(device)
                    report(
                        {
                            "stage": "ppo_resume",
                            "seed": seed_i,
                            "seed_status": loaded_status,
                        }
                    )
                else:
                    partial = load_seed_partial_checkpoint(
                        ckpt_dir, seed=seed_i, expected=dict(checkpoint_expected)
                    )
                    resume = None
                    if partial is not None:
                        resume = partial
                        report(
                            {
                                "stage": "ppo_resume",
                                "seed": seed_i,
                                "seed_status": "partial",
                                "steps_done": int(partial["steps_done"]),
                            }
                        )

                    def _on_episode(
                        _seed: int = seed_i,
                        _counters: dict[str, int] = counters,
                        **payload: Any,
                    ) -> None:
                        nonlocal ledger
                        ledger = record_episode_partial(
                            ckpt_dir,
                            seed=_seed,
                            device=device,
                            checkpoint_expected=checkpoint_expected,
                            counters=_counters,
                            ledger=ledger,
                            payload=payload,
                        )

                    if loaded_status not in {"trained", "validation_failed"}:
                        _log(f"[PPO] seed={seed_i} start device={device.type}")
                        train_ppo_discovery(
                            seed_policy,
                            _episode,
                            config=config,
                            seed=seed_i,
                            device=device,
                            resume=resume,
                            on_episode_complete=_on_episode,
                            progress=report,
                        )
                        save_seed_checkpoint(
                            ckpt_dir,
                            seed=seed_i,
                            policy=seed_policy,
                            metadata={
                                "experiment_id": experiment_id,
                                "status": "trained",
                                **dict(checkpoint_expected),
                            },
                        )
                        ledger = upsert_seed_row(
                            ledger,
                            seed_i,
                            status="trained",
                            steps_done=counters["steps_done"],
                            episode_index=counters["episode_index"],
                            update_index=counters["update_index"],
                            device=device.type,
                        )
                        write_seeds_ledger(ckpt_dir, ledger)
                val_metrics = eval_weeks(
                    seed_policy, val_weeks, snapshot, ohlcv, spy, scalers, config
                )
                save_seed_checkpoint(
                    ckpt_dir,
                    seed=seed_i,
                    policy=seed_policy,
                    metadata={
                        "experiment_id": experiment_id,
                        "status": "complete",
                        "val_cagr": val_metrics["cagr"],
                        "val_sharpe": val_metrics["sharpe"],
                        **dict(checkpoint_expected),
                    },
                )
                ledger = upsert_seed_row(
                    ledger,
                    seed_i,
                    status="complete",
                    val_cagr=val_metrics["cagr"],
                    val_sharpe=val_metrics["sharpe"],
                    device=device.type,
                )
                write_seeds_ledger(ckpt_dir, ledger)
                _log(
                    f"[PPO] seed={seed_i} val complete cagr={val_metrics['cagr']:.4f} "
                    f"sharpe={val_metrics['sharpe']:.4f} device={device.type}"
                )
            seed_policies[seed_i] = seed_policy
        except Exception as exc:
            last_error = exc
            if is_accelerator_out_of_memory(exc, device):
                fail_job_on_accelerator_oom(
                    exc,
                    seed=seed_i,
                    device=device,
                    directory=ckpt_dir,
                    ledger=ledger,
                    checkpoint_expected=checkpoint_expected,
                    progress=report,
                    steps_done=counters["steps_done"],
                    episode_index=counters["episode_index"],
                    update_index=counters["update_index"],
                )
            trained = (
                load_seed_checkpoint(
                    ckpt_dir, seed=seed_i, expected=dict(checkpoint_expected)
                )
                is not None
            )
            status = "validation_failed" if trained else "failed"
            error_row = {
                "status": status,
                "error_type": type(exc).__name__,
                "error": bound_error_message(exc),
                "device": device.type,
                **dict(checkpoint_expected),
            }
            write_seed_metadata(ckpt_dir, seed=seed_i, payload=error_row)
            if trained:
                save_seed_checkpoint(
                    ckpt_dir,
                    seed=seed_i,
                    policy=seed_policy,
                    metadata={"experiment_id": experiment_id, **error_row},
                )
            ledger = upsert_seed_row(ledger, seed_i, **error_row)
            write_seeds_ledger(ckpt_dir, ledger)
            logger.exception("ppo_discovery seed=%s failed", seed_i)
            continue

    declared = {int(seed) for seed in config.seeds}
    complete = {s: r for s, r in complete_seed_rows(ledger).items() if s in declared}
    failed = [seed for seed in failed_seed_ids(ledger) if seed in declared]
    seed_val = {seed: float(row["val_cagr"]) for seed, row in complete.items()}
    seed_sharpe = {seed: float(row["val_sharpe"]) for seed, row in complete.items()}
    if not seed_val:
        messages = [
            f"{seed}:{row.get('error_type')}:{row.get('error')}"
            for seed, row in (ledger.get("seeds") or {}).items()
            if int(seed) in failed
        ]
        raise PPODiscoveryError(
            "every ppo_discovery seed failed: " + "; ".join(messages)
        ) from last_error
    chosen = select_candidate_seed(seed_val, seed_sharpe)
    _log(
        f"[PPO] selected_seed={chosen} from complete={sorted(seed_val)} "
        f"failed={failed} device={device.type}"
    )
    return PPOSeedTrainingResult(
        selected_seed=chosen,
        selected_policy=seed_policies[chosen],
        seed_metrics={
            str(seed): {"val_cagr": seed_val[seed], "val_sharpe": seed_sharpe[seed]}
            for seed in seed_val
        },
        seed_aggregates={
            "val_cagr": aggregate_seed_metrics(seed_val),
            "val_sharpe": aggregate_seed_metrics(seed_sharpe),
            "n_seeds": len(seed_val),
        },
        failed_seeds=failed,
        ledger=ledger,
    )


__all__ = [
    "PPOSeedTrainingResult",
    "eval_weeks",
    "train_ppo_discovery_seeds",
    "week_logs",
]

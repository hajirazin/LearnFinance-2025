"""Universe-keyed local SAC storage classes."""

from datetime import UTC, datetime
from typing import Any

from brain_api.core.sac.config import SACConfig

from .artifacts import SACArtifacts
from .filesystem import SACFilesystemStorage


class SACHalalFilteredModelStorage(SACFilesystemStorage):
    """SAC artifacts trained on the monthly ``halal_filtered`` slate."""

    bucket_name = "sac_halal_filtered"


class SACHalalModelStorage(SACFilesystemStorage):
    """SAC artifacts trained on the legacy ``halal`` universe."""

    bucket_name = "sac_halal"


# Backward-compatible name used before universe-keyed buckets were introduced.
SACLocalStorage = SACHalalFilteredModelStorage


def create_sac_metadata(
    version: str,
    data_window_start: str,
    data_window_end: str,
    symbols: list[str],
    config: SACConfig,
    promoted: bool,
    prior_version: str | None,
    actor_loss: float,
    critic_loss: float,
    avg_episode_return: float,
    avg_episode_sharpe: float,
    eval_sharpe: float,
    eval_cagr: float,
    eval_max_drawdown: float,
    bucket_name: str = "sac_halal_filtered",
    failure_reasons: list[str] | None = None,
    training_seed: int | None = None,
    experiment_seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Create auditable metadata for a candidate or promoted SAC artifact."""
    metadata: dict[str, Any] = {
        "model_type": bucket_name,
        "version": version,
        "training_timestamp": datetime.now(UTC).isoformat(),
        "data_window": {"start": data_window_start, "end": data_window_end},
        "symbols": symbols,
        "config": config.to_dict(),
        "promoted": promoted,
        "prior_version": prior_version,
        "failure_reasons": list(failure_reasons or []),
        "metrics": {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "avg_episode_return": avg_episode_return,
            "avg_episode_sharpe": avg_episode_sharpe,
            "eval_sharpe": eval_sharpe,
            "eval_cagr": eval_cagr,
            "eval_max_drawdown": eval_max_drawdown,
        },
    }
    if training_seed is not None:
        metadata["training_seed"] = training_seed
    if experiment_seeds is not None:
        metadata["experiment_seeds"] = experiment_seeds
    return metadata


__all__ = [
    "SACArtifacts",
    "SACHalalFilteredModelStorage",
    "SACHalalModelStorage",
    "SACLocalStorage",
    "create_sac_metadata",
]

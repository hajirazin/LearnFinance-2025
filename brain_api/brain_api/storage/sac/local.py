"""Universe-keyed local SAC storage classes."""

from datetime import UTC, datetime
from typing import Any

from brain_api.core.portfolio_rl.state import (
    SAC_ASSET_FEATURE_NAMES,
    SAC_GLOBAL_FEATURE_NAMES,
    STATE_DIM,
)
from brain_api.core.sac.config import SACConfig

from .artifacts import (
    SAC_ACTION_DIM,
    SAC_ARCHITECTURE,
    SAC_MAX_ASSETS,
    SAC_SCHEMA_VERSION,
    SACArtifacts,
)
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
        "sac_schema_version": SAC_SCHEMA_VERSION,
        "architecture": SAC_ARCHITECTURE,
        "max_assets": SAC_MAX_ASSETS,
        "action_dim": SAC_ACTION_DIM,
        "state_dim": STATE_DIM,
        "asset_feature_names": list(SAC_ASSET_FEATURE_NAMES),
        "global_feature_names": list(SAC_GLOBAL_FEATURE_NAMES),
        "news_schema_version": 1,
        "finbert_revision": "4556d13015211d73dccd3fdd39d39232506f3e43",
        "version": version,
        "training_timestamp": datetime.now(UTC).isoformat(),
        "data_window": {"start": data_window_start, "end": data_window_end},
        "symbols": sorted(symbols),
        "symbol_to_slot": {symbol: slot for slot, symbol in enumerate(sorted(symbols))},
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

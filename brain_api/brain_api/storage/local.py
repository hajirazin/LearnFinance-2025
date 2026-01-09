"""Local filesystem storage for model artifacts.

This module re-exports from model-specific submodules for backward compatibility.
"""

# LSTM storage
# Shared utilities
from datetime import UTC

from brain_api.storage.base import DEFAULT_DATA_PATH
from brain_api.storage.lstm.local import (
    LocalModelStorage,  # Backward compatibility alias
    LSTMArtifacts,
    LSTMLocalStorage,
)
from brain_api.storage.metadata import create_training_metadata

# PatchTST storage
from brain_api.storage.patchtst.local import (
    PatchTSTArtifacts,
    PatchTSTModelStorage,
)

# PPO + LSTM storage
from brain_api.storage.ppo_lstm.local import (
    PPOLSTMArtifacts,
    PPOLSTMLocalStorage,
)

# PPO + PatchTST storage
from brain_api.storage.ppo_patchtst.local import (
    PPOPatchTSTArtifacts,
    PPOPatchTSTLocalStorage,
)

# SAC + LSTM storage
from brain_api.storage.sac_lstm.local import (
    SACLSTMArtifacts,
    SACLSTMLocalStorage,
    create_sac_lstm_metadata,
)

# SAC + PatchTST storage
from brain_api.storage.sac_patchtst.local import (
    SACPatchTSTArtifacts,
    SACPatchTSTLocalStorage,
    create_sac_patchtst_metadata,
)


def create_metadata(
    version: str,
    data_window_start: str,
    data_window_end: str,
    symbols: list[str],
    config,
    train_loss: float,
    val_loss: float,
    baseline_loss: float,
    promoted: bool,
    prior_version: str | None,
) -> dict:
    """Create metadata dict for LSTM training run (backward compatibility wrapper)."""
    return create_training_metadata(
        model_type="lstm",
        version=version,
        data_window_start=data_window_start,
        data_window_end=data_window_end,
        symbols=symbols,
        config_dict=config.to_dict(),
        train_loss=train_loss,
        val_loss=val_loss,
        baseline_loss=baseline_loss,
        promoted=promoted,
        prior_version=prior_version,
    )


def create_patchtst_metadata(
    version: str,
    data_window_start: str,
    data_window_end: str,
    symbols: list[str],
    config,
    train_loss: float,
    val_loss: float,
    baseline_loss: float,
    promoted: bool,
    prior_version: str | None,
) -> dict:
    """Create metadata dict for PatchTST training run (backward compatibility wrapper)."""
    return create_training_metadata(
        model_type="patchtst",
        version=version,
        data_window_start=data_window_start,
        data_window_end=data_window_end,
        symbols=symbols,
        config_dict=config.to_dict(),
        train_loss=train_loss,
        val_loss=val_loss,
        baseline_loss=baseline_loss,
        promoted=promoted,
        prior_version=prior_version,
    )


def create_ppo_lstm_metadata(
    version: str,
    data_window_start: str,
    data_window_end: str,
    symbols: list[str],
    config,
    promoted: bool,
    prior_version: str | None,
    # PPO-specific metrics
    policy_loss: float,
    value_loss: float,
    avg_episode_return: float,
    avg_episode_sharpe: float,
    eval_sharpe: float,
    eval_cagr: float,
    eval_max_drawdown: float,
) -> dict:
    """Create metadata dict for PPO + LSTM training run."""
    from datetime import datetime

    return {
        "model_type": "ppo_lstm",
        "version": version,
        "created_at": datetime.now(UTC).isoformat(),
        "data_window": {
            "start": data_window_start,
            "end": data_window_end,
        },
        "symbols": symbols,
        "config": config.to_dict(),
        "metrics": {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "avg_episode_return": avg_episode_return,
            "avg_episode_sharpe": avg_episode_sharpe,
            "eval_sharpe": eval_sharpe,
            "eval_cagr": eval_cagr,
            "eval_max_drawdown": eval_max_drawdown,
        },
        "promoted": promoted,
        "prior_version": prior_version,
    }


def create_ppo_patchtst_metadata(
    version: str,
    data_window_start: str,
    data_window_end: str,
    symbols: list[str],
    config,
    promoted: bool,
    prior_version: str | None,
    # PPO-specific metrics
    policy_loss: float,
    value_loss: float,
    avg_episode_return: float,
    avg_episode_sharpe: float,
    eval_sharpe: float,
    eval_cagr: float,
    eval_max_drawdown: float,
) -> dict:
    """Create metadata dict for PPO + PatchTST training run."""
    from datetime import datetime

    return {
        "model_type": "ppo_patchtst",
        "version": version,
        "created_at": datetime.now(UTC).isoformat(),
        "data_window": {
            "start": data_window_start,
            "end": data_window_end,
        },
        "symbols": symbols,
        "config": config.to_dict(),
        "metrics": {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "avg_episode_return": avg_episode_return,
            "avg_episode_sharpe": avg_episode_sharpe,
            "eval_sharpe": eval_sharpe,
            "eval_cagr": eval_cagr,
            "eval_max_drawdown": eval_max_drawdown,
        },
        "promoted": promoted,
        "prior_version": prior_version,
    }


__all__ = [
    # Shared
    "DEFAULT_DATA_PATH",
    # LSTM
    "LSTMArtifacts",
    "LSTMLocalStorage",
    "LocalModelStorage",  # Backward compatibility alias
    # PPO + LSTM
    "PPOLSTMArtifacts",
    "PPOLSTMLocalStorage",
    # PPO + PatchTST
    "PPOPatchTSTArtifacts",
    "PPOPatchTSTLocalStorage",
    # PatchTST
    "PatchTSTArtifacts",
    "PatchTSTModelStorage",
    # SAC + LSTM
    "SACLSTMArtifacts",
    "SACLSTMLocalStorage",
    # SAC + PatchTST
    "SACPatchTSTArtifacts",
    "SACPatchTSTLocalStorage",
    "create_metadata",
    "create_patchtst_metadata",
    "create_ppo_lstm_metadata",
    "create_ppo_patchtst_metadata",
    "create_sac_lstm_metadata",
    "create_sac_patchtst_metadata",
    "create_training_metadata",
]

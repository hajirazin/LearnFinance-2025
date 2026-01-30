"""Local filesystem storage for model artifacts.

This module re-exports from model-specific submodules for backward compatibility.
"""

# LSTM storage
# Shared utilities

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

# PPO storage (unified with dual forecasts)
from brain_api.storage.ppo.local import (
    PPOArtifacts,
    PPOLocalStorage,
    create_ppo_metadata,
)

# SAC storage (unified with dual forecasts)
from brain_api.storage.sac.local import (
    SACArtifacts,
    SACLocalStorage,
    create_sac_metadata,
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


__all__ = [
    # Shared
    "DEFAULT_DATA_PATH",
    # LSTM
    "LSTMArtifacts",
    "LSTMLocalStorage",
    "LocalModelStorage",  # Backward compatibility alias
    # PPO (unified with dual forecasts)
    "PPOArtifacts",
    "PPOLocalStorage",
    # PatchTST
    "PatchTSTArtifacts",
    "PatchTSTModelStorage",
    # SAC (unified with dual forecasts)
    "SACArtifacts",
    "SACLocalStorage",
    "create_metadata",
    "create_patchtst_metadata",
    "create_ppo_metadata",
    "create_sac_metadata",
    "create_training_metadata",
]

"""Local filesystem storage for model artifacts.

This module re-exports from model-specific submodules for backward compatibility.
"""

# LSTM storage
from brain_api.storage.lstm.local import (
    LSTMArtifacts,
    LSTMLocalStorage,
    LocalModelStorage,  # Backward compatibility alias
)

# PatchTST storage
from brain_api.storage.patchtst.local import (
    PatchTSTArtifacts,
    PatchTSTModelStorage,
)

# Shared utilities
from brain_api.storage.base import DEFAULT_DATA_PATH
from brain_api.storage.metadata import create_training_metadata


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
    # LSTM
    "LSTMArtifacts",
    "LSTMLocalStorage",
    "LocalModelStorage",  # Backward compatibility alias
    "create_metadata",
    # PatchTST
    "PatchTSTArtifacts",
    "PatchTSTModelStorage",
    "create_patchtst_metadata",
    # Shared
    "DEFAULT_DATA_PATH",
    "create_training_metadata",
]

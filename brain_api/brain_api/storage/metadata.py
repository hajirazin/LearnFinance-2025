"""Shared metadata utilities for model storage.

This module provides a unified metadata factory for all model types.
"""

from datetime import UTC, datetime
from typing import Any


def create_training_metadata(
    model_type: str,
    version: str,
    data_window_start: str,
    data_window_end: str,
    symbols: list[str],
    config_dict: dict[str, Any],
    train_loss: float,
    val_loss: float,
    baseline_loss: float,
    promoted: bool,
    prior_version: str | None,
) -> dict[str, Any]:
    """Create metadata dict for a training run.

    This is a unified factory that works for all model types (LSTM, PatchTST, etc.).

    Args:
        model_type: Model type identifier (e.g., "lstm", "patchtst")
        version: Version string
        data_window_start: Training data start date (ISO format)
        data_window_end: Training data end date (ISO format)
        symbols: List of symbols used for training
        config_dict: Model configuration as dictionary
        train_loss: Final training loss
        val_loss: Validation loss
        baseline_loss: Baseline (persistence) loss
        promoted: Whether this version was promoted to current
        prior_version: Previous current version (if any)

    Returns:
        Metadata dictionary
    """
    return {
        "model_type": model_type,
        "version": version,
        "training_timestamp": datetime.now(UTC).isoformat(),
        "data_window": {
            "start": data_window_start,
            "end": data_window_end,
        },
        "symbols": symbols,
        "config": config_dict,
        "metrics": {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "baseline_loss": baseline_loss,
        },
        "promoted": promoted,
        "prior_version": prior_version,
    }

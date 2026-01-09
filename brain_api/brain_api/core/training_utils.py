"""Shared training utilities for ML models.

This module contains common functions used by multiple model training pipelines.
"""

import torch


def get_device() -> torch.device:
    """Get the best available device for training.

    Priority:
    1. MPS (Apple Silicon GPU) - for M1/M2/M3 Macs
    2. CUDA (NVIDIA GPU)
    3. CPU (fallback)

    Returns:
        torch.device for the best available accelerator
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate_for_promotion(
    val_loss: float,
    baseline_loss: float,
    prior_val_loss: float | None,
) -> bool:
    """Decide whether to promote the new model to current.

    Promotion requires:
    1. Beat the baseline (persistence model)
    2. Beat the prior model (if one exists)

    Args:
        val_loss: Validation loss of new model
        baseline_loss: Loss of persistence baseline
        prior_val_loss: Validation loss of prior model (None if first model)

    Returns:
        True if model should be promoted
    """
    # Must beat baseline
    if val_loss >= baseline_loss:
        return False

    # Must beat prior (if exists)
    return prior_val_loss is None or val_loss < prior_val_loss

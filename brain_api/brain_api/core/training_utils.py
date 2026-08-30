"""Shared training utilities for ML models.

This module contains common functions used by multiple model training pipelines.
"""

import math
from pathlib import Path

import torch

from brain_api.core.training_health import ArtifactHealthCheck


class TrainingCancelledError(Exception):
    """Raised when training is cancelled by a shutdown event."""


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


def is_accelerator_out_of_memory(
    exc: BaseException,
    device: torch.device,
) -> bool:
    """Return true for Python, CUDA, or MPS out-of-memory failures."""
    if isinstance(exc, MemoryError):
        return True
    cuda_oom = getattr(torch.cuda, "OutOfMemoryError", None)
    if cuda_oom is not None and isinstance(exc, cuda_oom):
        return True
    if device.type == "mps" and isinstance(exc, RuntimeError):
        message = " ".join(str(exc).lower().split())
        return "out of memory" in message
    return False


# ---------------------------------------------------------------------------
# Forecaster artifact health check (always-promote-with-guardrails policy)
# ---------------------------------------------------------------------------

# Files every forecaster training run must persist on disk before the
# health check runs. Drifting from this list will cause the
# file-existence guardrails to silently pass when they shouldn't, so
# the constant lives next to the function rather than in a shared
# config to keep the change surface obvious.
_FORECASTER_REQUIRED_FILES: tuple[str, ...] = (
    "weights.pt",
    "feature_scaler.pkl",
    "config.json",
    "metadata.json",
)


def evaluate_forecaster_artifact_health(
    *,
    train_loss: float,
    val_loss: float,
    baseline_loss: float | None = None,
    artifact_dir: Path | None = None,
) -> ArtifactHealthCheck:
    """Run the forecaster (LSTM, PatchTST US, PatchTST India) guardrails.

    Replaces the prior ``evaluate_for_promotion(val_loss, prior_val_loss)``
    gate. The new policy is "always promote when guardrails pass" --
    the prior model's val_loss is no longer consulted because the
    universe rebuild + sliding validation window made it an
    apples-to-oranges baseline.

    Guardrails (each failure appends a stable, human-readable string):

    1. ``val_loss`` is finite AND ``> 0``
    2. ``train_loss`` is finite AND ``> 0``
    3. ``baseline_loss`` is finite AND ``> 0`` (skipped when ``None``;
       snapshot persist has no baseline)
    4-7. Each of ``weights.pt``, ``feature_scaler.pkl``, ``config.json``,
       and ``metadata.json`` exists under ``artifact_dir`` with a
       non-zero size (skipped when ``artifact_dir`` is ``None``;
       metrics-first snapshot persist writes files only after this
       check passes)

    Returns:
        :class:`ArtifactHealthCheck` whose ``is_healthy`` is the new
        promotion decision (main) or canonical-snapshot decision.
    """
    failure_reasons: list[str] = []

    if not math.isfinite(val_loss):
        failure_reasons.append("val_loss is not finite")
    elif val_loss <= 0:
        failure_reasons.append(f"val_loss must be > 0, got {val_loss}")

    if not math.isfinite(train_loss):
        failure_reasons.append("train_loss is not finite")
    elif train_loss <= 0:
        failure_reasons.append(f"train_loss must be > 0, got {train_loss}")

    if baseline_loss is not None:
        if not math.isfinite(baseline_loss):
            failure_reasons.append("baseline_loss is not finite")
        elif baseline_loss <= 0:
            failure_reasons.append(f"baseline_loss must be > 0, got {baseline_loss}")

    if artifact_dir is not None:
        for filename in _FORECASTER_REQUIRED_FILES:
            path = artifact_dir / filename
            if not path.exists() or path.stat().st_size <= 0:
                failure_reasons.append(f"{filename} missing or zero bytes")

    return ArtifactHealthCheck(
        is_healthy=not failure_reasons,
        failure_reasons=failure_reasons,
    )

"""SAC portfolio allocator with dual forecasts (LSTM + PatchTST).

This module provides SAC-based portfolio allocation using both LSTM
and PatchTST predicted weekly returns as forecast features.
"""

from brain_api.core.sac.config import (
    DEFAULT_SAC_CONFIG,
    SACConfig,
)
from brain_api.core.sac.inference import run_sac_inference
from brain_api.core.sac.training import (
    SACTrainingResult,
    TrainingData,
    build_training_data,
    finetune_sac,
    train_sac,
)
from brain_api.core.sac.version import compute_version

__all__ = [
    "DEFAULT_SAC_CONFIG",
    # Config
    "SACConfig",
    "SACTrainingResult",
    "TrainingData",
    "build_training_data",
    # Version
    "compute_version",
    "finetune_sac",
    # Inference
    "run_sac_inference",
    # Training
    "train_sac",
]

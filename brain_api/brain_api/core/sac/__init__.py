"""SAC portfolio allocator with PatchTST-only forecasts.

This module provides SAC-based portfolio allocation using PatchTST
predicted weekly returns as forecast features.
"""

from brain_api.core.sac.config import (
    DEFAULT_SAC_CONFIG,
    SACConfig,
    make_sac_config_for_n_stocks,
)
from brain_api.core.sac.experiment import (
    SAC_EXPERIMENT_SEEDS,
    SACCandidate,
    SACTrainingExperiment,
)
from brain_api.core.sac.inference import run_sac_inference
from brain_api.core.sac.training import (
    SACTrainingResult,
    TrainingData,
    build_training_data,
    train_sac,
)
from brain_api.core.sac.version import compute_version

__all__ = [
    "DEFAULT_SAC_CONFIG",
    "SAC_EXPERIMENT_SEEDS",
    "SACCandidate",
    # Config
    "SACConfig",
    "SACTrainingExperiment",
    "SACTrainingResult",
    "TrainingData",
    "build_training_data",
    # Version
    "compute_version",
    "make_sac_config_for_n_stocks",
    # Inference
    "run_sac_inference",
    # Training
    "train_sac",
]

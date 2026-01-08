"""SAC + PatchTST portfolio allocator.

This module provides SAC-based portfolio allocation using PatchTST
predicted weekly returns as the forecast feature.
"""

from brain_api.core.sac_patchtst.config import (
    SACPatchTSTConfig,
    DEFAULT_SAC_PATCHTST_CONFIG,
)
from brain_api.core.sac_patchtst.version import compute_version
from brain_api.core.sac_patchtst.training import (
    train_sac_patchtst,
    finetune_sac_patchtst,
    SACPatchTSTTrainingResult,
)
from brain_api.core.sac_patchtst.inference import run_sac_inference

# Re-export TrainingData and build_training_data from sac_lstm
# (same structure, different forecast source)
from brain_api.core.sac_lstm.training import TrainingData, build_training_data

__all__ = [
    # Config
    "SACPatchTSTConfig",
    "DEFAULT_SAC_PATCHTST_CONFIG",
    # Version
    "compute_version",
    # Training
    "train_sac_patchtst",
    "finetune_sac_patchtst",
    "SACPatchTSTTrainingResult",
    "TrainingData",
    "build_training_data",
    # Inference
    "run_sac_inference",
]


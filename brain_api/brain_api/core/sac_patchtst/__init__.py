"""SAC + PatchTST portfolio allocator.

This module provides SAC-based portfolio allocation using PatchTST
predicted weekly returns as the forecast feature.
"""

# Re-export TrainingData and build_training_data from sac_lstm
# (same structure, different forecast source)
from brain_api.core.sac_lstm.training import TrainingData, build_training_data
from brain_api.core.sac_patchtst.config import (
    DEFAULT_SAC_PATCHTST_CONFIG,
    SACPatchTSTConfig,
)
from brain_api.core.sac_patchtst.inference import run_sac_inference
from brain_api.core.sac_patchtst.training import (
    SACPatchTSTTrainingResult,
    finetune_sac_patchtst,
    train_sac_patchtst,
)
from brain_api.core.sac_patchtst.version import compute_version

__all__ = [
    "DEFAULT_SAC_PATCHTST_CONFIG",
    # Config
    "SACPatchTSTConfig",
    "SACPatchTSTTrainingResult",
    "TrainingData",
    "build_training_data",
    # Version
    "compute_version",
    "finetune_sac_patchtst",
    # Inference
    "run_sac_inference",
    # Training
    "train_sac_patchtst",
]


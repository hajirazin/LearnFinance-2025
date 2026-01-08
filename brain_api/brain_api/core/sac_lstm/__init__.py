"""SAC + LSTM portfolio allocator.

This module provides SAC-based portfolio allocation using LSTM
predicted weekly returns as the forecast feature.
"""

from brain_api.core.sac_lstm.config import (
    SACLSTMConfig,
    DEFAULT_SAC_LSTM_CONFIG,
)
from brain_api.core.sac_lstm.version import compute_version
from brain_api.core.sac_lstm.training import (
    train_sac_lstm,
    finetune_sac_lstm,
    SACLSTMTrainingResult,
    TrainingData,
    build_training_data,
)
from brain_api.core.sac_lstm.inference import run_sac_inference

__all__ = [
    # Config
    "SACLSTMConfig",
    "DEFAULT_SAC_LSTM_CONFIG",
    # Version
    "compute_version",
    # Training
    "train_sac_lstm",
    "finetune_sac_lstm",
    "SACLSTMTrainingResult",
    "TrainingData",
    "build_training_data",
    # Inference
    "run_sac_inference",
]


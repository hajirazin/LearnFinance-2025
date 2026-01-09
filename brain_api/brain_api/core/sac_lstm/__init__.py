"""SAC + LSTM portfolio allocator.

This module provides SAC-based portfolio allocation using LSTM
predicted weekly returns as the forecast feature.
"""

from brain_api.core.sac_lstm.config import (
    DEFAULT_SAC_LSTM_CONFIG,
    SACLSTMConfig,
)
from brain_api.core.sac_lstm.inference import run_sac_inference
from brain_api.core.sac_lstm.training import (
    SACLSTMTrainingResult,
    TrainingData,
    build_training_data,
    finetune_sac_lstm,
    train_sac_lstm,
)
from brain_api.core.sac_lstm.version import compute_version

__all__ = [
    "DEFAULT_SAC_LSTM_CONFIG",
    # Config
    "SACLSTMConfig",
    "SACLSTMTrainingResult",
    "TrainingData",
    "build_training_data",
    # Version
    "compute_version",
    "finetune_sac_lstm",
    # Inference
    "run_sac_inference",
    # Training
    "train_sac_lstm",
]


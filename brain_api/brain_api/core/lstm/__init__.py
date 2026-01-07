"""LSTM model module for weekly return prediction.

The LSTM predicts weekly returns (Mon open → Fri close), not daily prices.
This aligns with the RL agent's weekly decision horizon.
"""

# Config
from brain_api.core.lstm.config import DEFAULT_CONFIG, LSTMConfig

# Model
from brain_api.core.lstm.model import LSTMModel

# Version
from brain_api.core.lstm.version import compute_version

# Dataset
from brain_api.core.lstm.dataset import DatasetResult, build_dataset

# Training
from brain_api.core.lstm.training import TrainingResult, train_model_pytorch

# Inference
from brain_api.core.lstm.inference import (
    InferenceFeatures,
    SymbolPrediction,
    build_inference_features,
    run_inference,
)

# Re-export shared utilities for backward compatibility
from brain_api.core.inference_utils import (
    WeekBoundaries,
    compute_week_boundaries,
    extract_trading_weeks,
)
from brain_api.core.prices import load_prices_yfinance
from brain_api.core.training_utils import evaluate_for_promotion, get_device

__all__ = [
    # Config
    "LSTMConfig",
    "DEFAULT_CONFIG",
    # Model
    "LSTMModel",
    # Version
    "compute_version",
    # Dataset
    "DatasetResult",
    "build_dataset",
    # Training
    "TrainingResult",
    "train_model_pytorch",
    # Inference
    "InferenceFeatures",
    "SymbolPrediction",
    "build_inference_features",
    "run_inference",
    # Shared utilities (backward compat)
    "WeekBoundaries",
    "compute_week_boundaries",
    "extract_trading_weeks",
    "load_prices_yfinance",
    "get_device",
    "evaluate_for_promotion",
]



"""LSTM model module for weekly return prediction.

The LSTM predicts weekly returns (Mon open → Fri close), not daily prices.
This aligns with the RL agent's weekly decision horizon.
"""

# Config
# Re-export shared utilities for backward compatibility
from brain_api.core.inference_utils import (
    WeekBoundaries,
    compute_week_boundaries,
    compute_week_from_cutoff,
    extract_trading_weeks,
)
from brain_api.core.lstm.config import DEFAULT_CONFIG, LSTMConfig

# Dataset
from brain_api.core.lstm.dataset import DatasetResult, build_dataset

# Inference
from brain_api.core.lstm.inference import (
    InferenceFeatures,
    SymbolPrediction,
    build_inference_features,
    run_inference,
)

# Model
from brain_api.core.lstm.model import LSTMModel

# Training
from brain_api.core.lstm.training import TrainingResult, train_model_pytorch

# Version
from brain_api.core.lstm.version import compute_version
from brain_api.core.prices import load_prices_yfinance
from brain_api.core.training_utils import evaluate_for_promotion, get_device

__all__ = [
    "DEFAULT_CONFIG",
    # Dataset
    "DatasetResult",
    # Inference
    "InferenceFeatures",
    # Config
    "LSTMConfig",
    # Model
    "LSTMModel",
    "SymbolPrediction",
    # Training
    "TrainingResult",
    # Shared utilities (backward compat)
    "WeekBoundaries",
    "build_dataset",
    "build_inference_features",
    # Version
    "compute_version",
    "compute_week_boundaries",
    "compute_week_from_cutoff",
    "evaluate_for_promotion",
    "extract_trading_weeks",
    "get_device",
    "load_prices_yfinance",
    "run_inference",
    "train_model_pytorch",
]

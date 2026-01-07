"""PatchTST model module for multi-channel weekly return prediction.

PatchTST (Patch Time Series Transformer) with multi-channel support:
- Price features (OHLCV log returns)
- News sentiment features
- Fundamental ratios

The model predicts weekly returns aligned with the RL agent's weekly
decision horizon (Mon open → Fri close).
"""

# Config
from brain_api.core.patchtst.config import DEFAULT_CONFIG, PatchTSTConfig

# Version
from brain_api.core.patchtst.version import compute_version

# Data loaders
from brain_api.core.patchtst.data_loaders import (
    align_multivariate_data,
    load_historical_fundamentals,
    load_historical_news_sentiment,
)

# Dataset
from brain_api.core.patchtst.dataset import DatasetResult, build_dataset

# Training
from brain_api.core.patchtst.training import TrainingResult, train_model_pytorch

# Inference
from brain_api.core.patchtst.inference import (
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
    "PatchTSTConfig",
    "DEFAULT_CONFIG",
    # Version
    "compute_version",
    # Data loaders
    "load_historical_news_sentiment",
    "load_historical_fundamentals",
    "align_multivariate_data",
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


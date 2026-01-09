"""PatchTST model module for multi-channel weekly return prediction.

PatchTST (Patch Time Series Transformer) with multi-channel support:
- Price features (OHLCV log returns)
- News sentiment features
- Fundamental ratios

The model predicts weekly returns aligned with the RL agent's weekly
decision horizon (Mon open → Fri close).
"""

# Config
# Re-export shared utilities for backward compatibility
from brain_api.core.inference_utils import (
    WeekBoundaries,
    compute_week_boundaries,
    extract_trading_weeks,
)
from brain_api.core.patchtst.config import DEFAULT_CONFIG, PatchTSTConfig

# Data loaders
from brain_api.core.patchtst.data_loaders import (
    align_multivariate_data,
    load_historical_fundamentals,
    load_historical_news_sentiment,
)

# Dataset
from brain_api.core.patchtst.dataset import DatasetResult, build_dataset

# Inference
from brain_api.core.patchtst.inference import (
    InferenceFeatures,
    SymbolPrediction,
    build_inference_features,
    run_inference,
)

# Training
from brain_api.core.patchtst.training import TrainingResult, train_model_pytorch

# Version
from brain_api.core.patchtst.version import compute_version
from brain_api.core.prices import load_prices_yfinance
from brain_api.core.training_utils import evaluate_for_promotion, get_device

__all__ = [
    "DEFAULT_CONFIG",
    # Dataset
    "DatasetResult",
    # Inference
    "InferenceFeatures",
    # Config
    "PatchTSTConfig",
    "SymbolPrediction",
    # Training
    "TrainingResult",
    # Shared utilities (backward compat)
    "WeekBoundaries",
    "align_multivariate_data",
    "build_dataset",
    "build_inference_features",
    # Version
    "compute_version",
    "compute_week_boundaries",
    "evaluate_for_promotion",
    "extract_trading_weeks",
    "get_device",
    "load_historical_fundamentals",
    # Data loaders
    "load_historical_news_sentiment",
    "load_prices_yfinance",
    "run_inference",
    "train_model_pytorch",
]


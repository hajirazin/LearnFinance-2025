"""PatchTST model module for 5-channel OHLCV multi-task prediction.

PatchTST (Patch Time Series Transformer) with channel-independent shared weights.
5-channel OHLCV input (open_ret, high_ret, low_ret, close_ret, volume_ret).
Multi-task loss on ALL 5 channels. Direct 5-day prediction with RevIN normalization.

At inference, close_ret predictions are extracted for weekly return calculation.
"""

# Config
# Re-export shared utilities for backward compatibility
from brain_api.core.inference_utils import (
    WeekBoundaries,
    compute_week_boundaries,
    extract_trading_weeks,
)
from brain_api.core.patchtst.config import DEFAULT_CONFIG, PatchTSTConfig

# Data loaders (OHLCV-only; load_historical_* remain in data_loaders for other consumers)
from brain_api.core.patchtst.data_loaders import align_multivariate_data

# Dataset
from brain_api.core.patchtst.dataset import DatasetResult, build_dataset

# Inference
from brain_api.core.patchtst.inference import (
    BatchInferenceResult,
    InferenceFeatures,
    SymbolPrediction,
    build_inference_features,
    run_batch_inference,
    run_inference,
)
from brain_api.core.patchtst.score_validation import (
    validate_and_collect_finite_scores,
)

# Training
from brain_api.core.patchtst.training import TrainingResult, train_model_pytorch

# Version
from brain_api.core.patchtst.version import compute_version
from brain_api.core.prices import load_prices_yfinance
from brain_api.core.training_utils import evaluate_for_promotion, get_device

__all__ = [
    "DEFAULT_CONFIG",
    "BatchInferenceResult",
    "DatasetResult",
    "InferenceFeatures",
    "PatchTSTConfig",
    "SymbolPrediction",
    "TrainingResult",
    "WeekBoundaries",
    "align_multivariate_data",
    "build_dataset",
    "build_inference_features",
    "compute_version",
    "compute_week_boundaries",
    "evaluate_for_promotion",
    "extract_trading_weeks",
    "get_device",
    "load_prices_yfinance",
    "run_batch_inference",
    "run_inference",
    "train_model_pytorch",
    "validate_and_collect_finite_scores",
]

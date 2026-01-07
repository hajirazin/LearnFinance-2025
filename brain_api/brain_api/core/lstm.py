"""Backward compatibility re-export.

The LSTM module has been split into the brain_api.core.lstm package.
This module re-exports for backward compatibility.
"""

from brain_api.core.lstm import (
    DEFAULT_CONFIG,
    DatasetResult,
    InferenceFeatures,
    LSTMConfig,
    LSTMModel,
    SymbolPrediction,
    TrainingResult,
    WeekBoundaries,
    build_dataset,
    build_inference_features,
    compute_version,
    compute_week_boundaries,
    evaluate_for_promotion,
    extract_trading_weeks,
    get_device,
    load_prices_yfinance,
    run_inference,
    train_model_pytorch,
)

__all__ = [
    "LSTMConfig",
    "DEFAULT_CONFIG",
    "LSTMModel",
    "compute_version",
    "DatasetResult",
    "build_dataset",
    "TrainingResult",
    "train_model_pytorch",
    "InferenceFeatures",
    "SymbolPrediction",
    "build_inference_features",
    "run_inference",
    "WeekBoundaries",
    "compute_week_boundaries",
    "extract_trading_weeks",
    "load_prices_yfinance",
    "get_device",
    "evaluate_for_promotion",
]

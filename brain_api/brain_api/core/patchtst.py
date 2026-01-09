"""Backward compatibility re-export.

The PatchTST module has been split into the brain_api.core.patchtst package.
This module re-exports for backward compatibility.
"""

from brain_api.core.patchtst import (
    DEFAULT_CONFIG,
    DatasetResult,
    InferenceFeatures,
    PatchTSTConfig,
    SymbolPrediction,
    TrainingResult,
    WeekBoundaries,
    align_multivariate_data,
    build_dataset,
    build_inference_features,
    compute_version,
    compute_week_boundaries,
    evaluate_for_promotion,
    extract_trading_weeks,
    get_device,
    load_historical_fundamentals,
    load_historical_news_sentiment,
    load_prices_yfinance,
    run_inference,
    train_model_pytorch,
)

__all__ = [
    "DEFAULT_CONFIG",
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
    "load_historical_fundamentals",
    "load_historical_news_sentiment",
    "load_prices_yfinance",
    "run_inference",
    "train_model_pytorch",
]

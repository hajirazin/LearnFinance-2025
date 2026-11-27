"""Backward compatibility re-export.

The PatchTST module has been split into the brain_api.core.patchtst package.
This module re-exports for backward compatibility.
"""

from brain_api.core.patchtst import (
    DEFAULT_CONFIG,
    InferenceFeatures,
    PatchTSTConfig,
    SymbolPrediction,
    WeekBoundaries,
    align_multivariate_data,
    build_inference_features,
    compute_version,
    compute_week_boundaries,
    extract_trading_weeks,
    get_device,
    load_prices_yfinance,
    run_inference,
)

__all__ = [
    "DEFAULT_CONFIG",
    "InferenceFeatures",
    "PatchTSTConfig",
    "SymbolPrediction",
    "WeekBoundaries",
    "align_multivariate_data",
    "build_inference_features",
    "compute_version",
    "compute_week_boundaries",
    "extract_trading_weeks",
    "get_device",
    "load_prices_yfinance",
    "run_inference",
]

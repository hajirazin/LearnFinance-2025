"""PatchTST inference, versioning, and historical checkpoint utilities.

Production US and India training uses the weekly regression modules. Legacy
feature helpers remain for historical checkpoint compatibility, not retraining.
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

# Version
from brain_api.core.patchtst.version import compute_version
from brain_api.core.prices import load_close_prices_yfinance as load_prices_yfinance
from brain_api.core.training_utils import get_device

__all__ = [
    "DEFAULT_CONFIG",
    "BatchInferenceResult",
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
    "run_batch_inference",
    "run_inference",
    "validate_and_collect_finite_scores",
]

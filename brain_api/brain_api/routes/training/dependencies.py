"""Shared dependency injection for training endpoints.

Universe selection now lives in the per-bucket registry
(``brain_api.core.model_buckets``) -- training endpoints take the
``universe`` from the request body and resolve symbols in-process via
``BucketConfig.symbols_resolver``. The old env-var dispatchers
(``get_forecaster_training_symbols`` / ``get_rl_training_symbols`` /
``get_top15_symbols``) were removed because they could not support two
parallel workflows hitting the same endpoint with different universes.
"""

from collections.abc import Callable
from typing import Any

from brain_api.core.lstm import (
    DEFAULT_CONFIG,
    DatasetResult,
    LSTMConfig,
    TrainingResult,
    build_dataset,
    load_prices_yfinance,
    train_model_pytorch,
)
from brain_api.core.patchtst import (
    DEFAULT_CONFIG as PATCHTST_DEFAULT_CONFIG,
)
from brain_api.core.patchtst import (
    DatasetResult as PatchTSTDatasetResult,
)
from brain_api.core.patchtst import PatchTSTConfig
from brain_api.core.patchtst import (
    TrainingResult as PatchTSTTrainingResult,
)
from brain_api.core.patchtst import (
    build_dataset as patchtst_build_dataset,
)
from brain_api.core.patchtst import (
    load_prices_yfinance as patchtst_load_prices,
)
from brain_api.core.patchtst import (
    train_model_pytorch as patchtst_train_model,
)
from brain_api.core.sac import DEFAULT_SAC_CONFIG, SACConfig
from brain_api.storage.forecaster_snapshots import SnapshotLocalStorage
from brain_api.storage.local import (
    LSTMHalalNewModelStorage,
    PatchTSTHalalNewModelStorage,
    PatchTSTNiftyShariah500ModelStorage,
)
from brain_api.storage.sac import SACHalalFilteredModelStorage

# ============================================================================
# Type aliases for dependency injection
# ============================================================================

# LSTM types
PriceLoader = Callable[[list[str], Any, Any], dict]
DatasetBuilder = Callable[[dict, LSTMConfig], DatasetResult]
Trainer = Callable[[Any, Any, Any, LSTMConfig], TrainingResult]

# PatchTST types
PatchTSTPriceLoader = Callable[[list[str], Any, Any], dict]
PatchTSTDatasetBuilder = Callable[[dict, dict, PatchTSTConfig], PatchTSTDatasetResult]
PatchTSTTrainer = Callable[[Any, Any, Any, PatchTSTConfig], PatchTSTTrainingResult]


def snapshots_available(forecaster_type: str) -> bool:
    """Check if forecaster snapshots are available for walk-forward inference.

    Args:
        forecaster_type: "lstm" or "patchtst"

    Returns:
        True if at least one snapshot exists
    """
    storage = SnapshotLocalStorage(forecaster_type)
    snapshots = storage.list_snapshots()
    return len(snapshots) > 0


# ============================================================================
# LSTM dependencies
# ============================================================================


def get_storage() -> LSTMHalalNewModelStorage:
    """Get the LSTM model storage instance for the halal_new bucket.

    Retained for tests that still inject the LSTM storage via FastAPI
    ``Depends`` overrides; production code paths now look up the
    storage via the bucket registry inside the endpoint.
    """
    return LSTMHalalNewModelStorage()


def get_config() -> LSTMConfig:
    """Get LSTM training configuration."""
    return DEFAULT_CONFIG


def get_price_loader() -> PriceLoader:
    """Get the price loading function."""
    return load_prices_yfinance


def get_dataset_builder() -> DatasetBuilder:
    """Get the dataset building function."""
    return build_dataset


def get_trainer() -> Trainer:
    """Get the training function."""
    return train_model_pytorch


# ============================================================================
# PatchTST dependencies
# ============================================================================


def get_patchtst_storage() -> PatchTSTHalalNewModelStorage:
    """Get the PatchTST model storage instance for the halal_new bucket."""
    return PatchTSTHalalNewModelStorage()


def get_patchtst_config() -> PatchTSTConfig:
    """Get PatchTST training configuration."""
    return PATCHTST_DEFAULT_CONFIG


def get_patchtst_price_loader() -> PatchTSTPriceLoader:
    """Get the price loading function for PatchTST."""
    return patchtst_load_prices


def get_patchtst_dataset_builder() -> PatchTSTDatasetBuilder:
    """Get the dataset building function."""
    return patchtst_build_dataset


def get_patchtst_trainer() -> PatchTSTTrainer:
    """Get the training function."""
    return patchtst_train_model


# ============================================================================
# PatchTST India dependencies
# ============================================================================


def get_patchtst_india_storage() -> PatchTSTNiftyShariah500ModelStorage:
    """Get the India PatchTST model storage instance for the nifty_shariah_500 bucket."""
    return PatchTSTNiftyShariah500ModelStorage()


# ============================================================================
# SAC dependencies (unified with dual forecasts)
# ============================================================================


def get_sac_storage() -> SACHalalFilteredModelStorage:
    """Get the SAC storage instance for the halal_filtered bucket."""
    return SACHalalFilteredModelStorage()


def get_sac_config() -> SACConfig:
    """Get SAC configuration."""
    return DEFAULT_SAC_CONFIG

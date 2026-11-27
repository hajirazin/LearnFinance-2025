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
from brain_api.core.patchtst.config import (
    INDIA_WEEKLY_CONFIG,
    US_WEEKLY_CONFIG,
    PatchTSTConfig,
)
from brain_api.core.patchtst.weekly_training import (
    WeeklyPatchTSTResult,
    fit_weekly_patchtst,
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
WeeklyPatchTSTTrainer = Callable[..., WeeklyPatchTSTResult]


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
    return US_WEEKLY_CONFIG


def get_india_patchtst_config() -> PatchTSTConfig:
    return INDIA_WEEKLY_CONFIG


def get_patchtst_price_loader() -> PatchTSTPriceLoader:
    """Get the price loading function for PatchTST."""
    from brain_api.core.prices import load_prices_yfinance

    def load(symbols, start, end):
        return load_prices_yfinance(
            symbols, start, end, required_columns=("close", "volume")
        )

    return load


def get_weekly_patchtst_trainer() -> WeeklyPatchTSTTrainer:
    """Get the weekly fitter used by main and annual training."""
    return fit_weekly_patchtst


# ============================================================================
# PatchTST India dependencies
# ============================================================================


def get_patchtst_india_storage() -> PatchTSTNiftyShariah500ModelStorage:
    """Get the India PatchTST model storage instance for the nifty_shariah_500 bucket."""
    return PatchTSTNiftyShariah500ModelStorage()


# ============================================================================
# SAC dependencies (PatchTST forecast features)
# ============================================================================


def get_sac_storage() -> SACHalalFilteredModelStorage:
    """Get the SAC storage instance for the halal_filtered bucket."""
    return SACHalalFilteredModelStorage()


def get_sac_config() -> SACConfig:
    """Get SAC configuration."""
    return DEFAULT_SAC_CONFIG

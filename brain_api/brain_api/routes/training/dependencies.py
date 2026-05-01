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
from brain_api.universe import get_halal_universe

# ``get_halal_universe`` is still used by the ETL symbol resolver below;
# imports kept for backward compatibility.
_ = get_halal_universe

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


# ============================================================================
# ETL symbol resolver (separate from training; ETL_UNIVERSE env still in use)
# ============================================================================


def get_etl_symbols() -> list[str]:
    """Get symbols for ETL pipelines based on config.

    Reads ``ETL_UNIVERSE`` env var to determine which universe to use.
    Default is ``UniverseType.HALAL_FILTERED``. Unlike the training
    pipelines, ETL is not part of the universe-keyed bucket registry
    -- it remains env-var driven because there is no concurrent A/B
    requirement for the ETL refresh job.
    """
    from brain_api.core.config import UniverseType, get_etl_universe

    universe_type = get_etl_universe()

    if universe_type == UniverseType.SP500:
        from brain_api.universe.sp500 import get_sp500_symbols

        return get_sp500_symbols()
    elif universe_type == UniverseType.HALAL_NEW:
        from brain_api.universe.halal_new import get_halal_new_symbols

        return get_halal_new_symbols()
    elif universe_type == UniverseType.HALAL_FILTERED:
        from brain_api.universe.halal_filtered import get_halal_filtered_symbols

        return get_halal_filtered_symbols()
    elif universe_type == UniverseType.NIFTY_SHARIAH_500:
        from brain_api.universe.nifty_shariah_500 import get_nifty_shariah_500_symbols

        return get_nifty_shariah_500_symbols()
    else:  # Default: HALAL
        universe = get_halal_universe()
        return [stock["symbol"] for stock in universe["stocks"]]


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

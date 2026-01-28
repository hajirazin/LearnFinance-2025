"""Shared dependency injection for training endpoints."""

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
from brain_api.core.patchtst import (
    PatchTSTConfig,
    align_multivariate_data,
    load_historical_fundamentals,
    load_historical_news_sentiment,
)
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
from brain_api.core.ppo_lstm import DEFAULT_PPO_LSTM_CONFIG, PPOLSTMConfig
from brain_api.core.ppo_patchtst import DEFAULT_PPO_PATCHTST_CONFIG, PPOPatchTSTConfig
from brain_api.core.sac_lstm import DEFAULT_SAC_LSTM_CONFIG, SACLSTMConfig
from brain_api.core.sac_patchtst import DEFAULT_SAC_PATCHTST_CONFIG, SACPatchTSTConfig
from brain_api.storage.forecaster_snapshots import SnapshotLocalStorage
from brain_api.storage.local import (
    LocalModelStorage,
    PatchTSTModelStorage,
    PPOLSTMLocalStorage,
    PPOPatchTSTLocalStorage,
    SACLSTMLocalStorage,
    SACPatchTSTLocalStorage,
)
from brain_api.universe import get_halal_universe

# ============================================================================
# Type aliases for dependency injection
# ============================================================================

# LSTM types
PriceLoader = Callable[[list[str], Any, Any], dict]
DatasetBuilder = Callable[[dict, LSTMConfig], DatasetResult]
Trainer = Callable[[Any, Any, Any, LSTMConfig], TrainingResult]

# PatchTST types
PatchTSTPriceLoader = Callable[[list[str], Any, Any], dict]
PatchTSTNewsLoader = Callable[[list[str], Any, Any], dict]
PatchTSTFundamentalsLoader = Callable[[list[str], Any, Any], dict]
PatchTSTDataAligner = Callable[[dict, dict, dict, PatchTSTConfig], dict]
PatchTSTDatasetBuilder = Callable[[dict, dict, PatchTSTConfig], PatchTSTDatasetResult]
PatchTSTTrainer = Callable[[Any, Any, Any, PatchTSTConfig], PatchTSTTrainingResult]


# ============================================================================
# Shared dependencies
# ============================================================================


def get_symbols() -> list[str]:
    """Get symbols for training from halal universe."""
    universe = get_halal_universe()
    return [stock["symbol"] for stock in universe["stocks"]]


def get_lstm_training_symbols() -> list[str]:
    """Get symbols for LSTM training based on config.

    Reads LSTM_TRAIN_UNIVERSE env var to determine which universe to use.
    Default is UniverseType.HALAL for backward compatibility.

    Returns:
        List of symbols for LSTM training
    """
    from brain_api.core.config import UniverseType, get_lstm_train_universe

    universe_type = get_lstm_train_universe()

    if universe_type == UniverseType.SP500:
        from brain_api.universe.sp500 import get_sp500_symbols

        return get_sp500_symbols()
    else:  # Default: HALAL
        universe = get_halal_universe()
        return [stock["symbol"] for stock in universe["stocks"]]


def get_top15_symbols() -> list[str]:
    """Get top 15 symbols by liquidity from halal universe."""
    universe = get_halal_universe()
    stocks = universe["stocks"][:15]
    return [stock["symbol"] for stock in stocks]


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


def get_storage() -> LocalModelStorage:
    """Get the model storage instance."""
    return LocalModelStorage()


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


def get_patchtst_storage() -> PatchTSTModelStorage:
    """Get the PatchTST model storage instance."""
    return PatchTSTModelStorage()


def get_patchtst_config() -> PatchTSTConfig:
    """Get PatchTST training configuration."""
    return PATCHTST_DEFAULT_CONFIG


def get_patchtst_price_loader() -> PatchTSTPriceLoader:
    """Get the price loading function for PatchTST."""
    return patchtst_load_prices


def get_patchtst_news_loader() -> PatchTSTNewsLoader:
    """Get the news sentiment loading function."""
    return load_historical_news_sentiment


def get_patchtst_fundamentals_loader() -> PatchTSTFundamentalsLoader:
    """Get the fundamentals loading function."""
    return load_historical_fundamentals


def get_patchtst_data_aligner() -> PatchTSTDataAligner:
    """Get the data alignment function."""
    return align_multivariate_data


def get_patchtst_dataset_builder() -> PatchTSTDatasetBuilder:
    """Get the dataset building function."""
    return patchtst_build_dataset


def get_patchtst_trainer() -> PatchTSTTrainer:
    """Get the training function."""
    return patchtst_train_model


# ============================================================================
# PPO dependencies
# ============================================================================


def get_ppo_lstm_storage() -> PPOLSTMLocalStorage:
    """Get the PPO + LSTM storage instance."""
    return PPOLSTMLocalStorage()


def get_ppo_lstm_config() -> PPOLSTMConfig:
    """Get PPO + LSTM configuration."""
    return DEFAULT_PPO_LSTM_CONFIG


def get_ppo_patchtst_storage() -> PPOPatchTSTLocalStorage:
    """Get the PPO + PatchTST storage instance."""
    return PPOPatchTSTLocalStorage()


def get_ppo_patchtst_config() -> PPOPatchTSTConfig:
    """Get PPO + PatchTST configuration."""
    return DEFAULT_PPO_PATCHTST_CONFIG


# ============================================================================
# SAC dependencies
# ============================================================================


def get_sac_lstm_storage() -> SACLSTMLocalStorage:
    """Get the SAC + LSTM storage instance."""
    return SACLSTMLocalStorage()


def get_sac_lstm_config() -> SACLSTMConfig:
    """Get SAC + LSTM configuration."""
    return DEFAULT_SAC_LSTM_CONFIG


def get_sac_patchtst_storage() -> SACPatchTSTLocalStorage:
    """Get the SAC + PatchTST storage instance."""
    return SACPatchTSTLocalStorage()


def get_sac_patchtst_config() -> SACPatchTSTConfig:
    """Get SAC + PatchTST configuration."""
    return DEFAULT_SAC_PATCHTST_CONFIG

"""Training endpoints for ML models.

This module provides training endpoints for various model types:
- LSTM: Pure price-based weekly return prediction
- PatchTST: OHLCV 5-channel weekly return prediction
- PPO: Portfolio allocator using dual forecasts (LSTM + PatchTST)
- SAC: Portfolio allocator using dual forecasts (LSTM + PatchTST)
"""

from fastapi import APIRouter

# Re-export dependencies for backward compatibility
from .dependencies import (
    get_config,
    get_dataset_builder,
    get_forecaster_training_symbols,
    get_patchtst_config,
    get_patchtst_data_aligner,
    get_patchtst_dataset_builder,
    get_patchtst_fundamentals_loader,
    get_patchtst_news_loader,
    get_patchtst_price_loader,
    get_patchtst_storage,
    get_patchtst_trainer,
    get_ppo_config,
    get_ppo_storage,
    get_price_loader,
    get_rl_training_symbols,
    get_sac_config,
    get_sac_storage,
    get_storage,
    get_top15_symbols,
    get_trainer,
    snapshots_available,
)

# Re-export internal functions for backward compatibility
from .lstm import _backfill_lstm_snapshots
from .lstm import router as lstm_router

# Re-export response models for backward compatibility
from .models import (
    LSTMTrainResponse,
    PatchTSTTrainResponse,
    PPOTrainResponse,
    SACTrainResponse,
)
from .patchtst import _backfill_patchtst_snapshots
from .patchtst import router as patchtst_router
from .ppo import router as ppo_router
from .sac import router as sac_router

# Backward compat alias for _snapshots_available
_snapshots_available = snapshots_available

# Re-export SnapshotLocalStorage for test patching compatibility
from brain_api.storage.forecaster_snapshots import SnapshotLocalStorage

# Create combined router
router = APIRouter()

# Include all sub-routers
router.include_router(lstm_router)
router.include_router(patchtst_router)
router.include_router(ppo_router)
router.include_router(sac_router)

__all__ = [
    # Response models
    "LSTMTrainResponse",
    "PPOTrainResponse",
    "PatchTSTTrainResponse",
    "SACTrainResponse",
    "SnapshotLocalStorage",  # Re-exported for test patching compatibility
    "_backfill_lstm_snapshots",
    "_backfill_patchtst_snapshots",
    "_snapshots_available",
    # Dependencies
    "get_config",
    "get_dataset_builder",
    "get_forecaster_training_symbols",
    "get_patchtst_config",
    "get_patchtst_data_aligner",
    "get_patchtst_dataset_builder",
    "get_patchtst_fundamentals_loader",
    "get_patchtst_news_loader",
    "get_patchtst_price_loader",
    "get_patchtst_storage",
    "get_patchtst_trainer",
    "get_ppo_config",
    "get_ppo_storage",
    "get_price_loader",
    "get_rl_training_symbols",
    "get_sac_config",
    "get_sac_storage",
    "get_storage",
    "get_top15_symbols",
    "get_trainer",
    "router",
    # Backward compat exports
    "snapshots_available",
]

"""Training endpoints for ML models.

This module provides training endpoints for various model types:
- LSTM: Pure price-based weekly return prediction
- PatchTST: Multi-signal weekly return prediction
- PPO + LSTM: Portfolio allocator using LSTM forecasts
- PPO + PatchTST: Portfolio allocator using PatchTST forecasts
- SAC + LSTM: Portfolio allocator using LSTM forecasts (SAC algorithm)
- SAC + PatchTST: Portfolio allocator using PatchTST forecasts (SAC algorithm)
"""

from fastapi import APIRouter

from .lstm import router as lstm_router
from .patchtst import router as patchtst_router
from .ppo_lstm import router as ppo_lstm_router
from .ppo_patchtst import router as ppo_patchtst_router
from .sac_lstm import router as sac_lstm_router
from .sac_patchtst import router as sac_patchtst_router

# Re-export response models for backward compatibility
from .models import (
    LSTMTrainResponse,
    PatchTSTTrainResponse,
    PPOLSTMTrainResponse,
    PPOPatchTSTTrainResponse,
    SACLSTMTrainResponse,
    SACPatchTSTTrainResponse,
)

# Re-export dependencies for backward compatibility
from .dependencies import (
    get_config,
    get_dataset_builder,
    get_patchtst_config,
    get_patchtst_data_aligner,
    get_patchtst_dataset_builder,
    get_patchtst_fundamentals_loader,
    get_patchtst_news_loader,
    get_patchtst_price_loader,
    get_patchtst_storage,
    get_patchtst_trainer,
    get_ppo_lstm_config,
    get_ppo_lstm_storage,
    get_ppo_patchtst_config,
    get_ppo_patchtst_storage,
    get_price_loader,
    get_sac_lstm_config,
    get_sac_lstm_storage,
    get_sac_patchtst_config,
    get_sac_patchtst_storage,
    get_storage,
    get_symbols,
    get_top15_symbols,
    get_trainer,
    snapshots_available,
)

# Re-export internal functions for backward compatibility
from .lstm import _backfill_lstm_snapshots
from .patchtst import _backfill_patchtst_snapshots

# Backward compat alias for _snapshots_available
_snapshots_available = snapshots_available

# Re-export SnapshotLocalStorage for test patching compatibility
from brain_api.storage.forecaster_snapshots import SnapshotLocalStorage

# Create combined router
router = APIRouter()

# Include all sub-routers
router.include_router(lstm_router)
router.include_router(patchtst_router)
router.include_router(ppo_lstm_router)
router.include_router(ppo_patchtst_router)
router.include_router(sac_lstm_router)
router.include_router(sac_patchtst_router)

__all__ = [
    "router",
    # Response models
    "LSTMTrainResponse",
    "PatchTSTTrainResponse",
    "PPOLSTMTrainResponse",
    "PPOPatchTSTTrainResponse",
    "SACLSTMTrainResponse",
    "SACPatchTSTTrainResponse",
    # Dependencies
    "get_config",
    "get_dataset_builder",
    "get_patchtst_config",
    "get_patchtst_data_aligner",
    "get_patchtst_dataset_builder",
    "get_patchtst_fundamentals_loader",
    "get_patchtst_news_loader",
    "get_patchtst_price_loader",
    "get_patchtst_storage",
    "get_patchtst_trainer",
    "get_ppo_lstm_config",
    "get_ppo_lstm_storage",
    "get_ppo_patchtst_config",
    "get_ppo_patchtst_storage",
    "get_price_loader",
    "get_sac_lstm_config",
    "get_sac_lstm_storage",
    "get_sac_patchtst_config",
    "get_sac_patchtst_storage",
    "get_storage",
    "get_symbols",
    "get_top15_symbols",
    "get_trainer",
    # Backward compat exports
    "snapshots_available",
    "_snapshots_available",
    "_backfill_lstm_snapshots",
    "_backfill_patchtst_snapshots",
]


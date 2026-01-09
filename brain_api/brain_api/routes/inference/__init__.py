"""Inference endpoints for ML models.

This module provides inference endpoints for various model types:
- LSTM: Pure price-based weekly return prediction
- PatchTST: Multi-signal weekly return prediction
- PPO + LSTM: Portfolio allocator using LSTM forecasts
- PPO + PatchTST: Portfolio allocator using PatchTST forecasts
- SAC + LSTM: Portfolio allocator using LSTM forecasts (SAC algorithm)
- SAC + PatchTST: Portfolio allocator using PatchTST forecasts (SAC algorithm)
"""

from fastapi import APIRouter

# Re-export dependencies for backward compatibility
from .dependencies import (
    get_as_of_date,
    get_patchtst_as_of_date,
    get_patchtst_storage,
    get_ppo_lstm_as_of_date,
    get_ppo_lstm_storage,
    get_ppo_patchtst_as_of_date,
    get_ppo_patchtst_storage,
    get_price_loader,
    get_sac_lstm_as_of_date,
    get_sac_lstm_storage,
    get_sac_patchtst_as_of_date,
    get_sac_patchtst_storage,
    get_sentiment_parquet_path,
    get_storage,
    get_week_boundary_computer,
)

# Re-export helpers for backward compatibility
from .helpers import (
    _load_model_artifacts,
    _load_model_artifacts_generic,
    _load_patchtst_model_artifacts,
    _sort_patchtst_predictions,
    _sort_predictions,
)
from .lstm import router as lstm_router

# Re-export response models for backward compatibility
from .models import (
    LSTMInferenceRequest,
    LSTMInferenceResponse,
    PatchTSTInferenceRequest,
    PatchTSTInferenceResponse,
    PortfolioSnapshot,
    Position,
    PPOLSTMInferenceRequest,
    PPOLSTMInferenceResponse,
    PPOPatchTSTInferenceRequest,
    PPOPatchTSTInferenceResponse,
    SACLSTMInferenceRequest,
    SACLSTMInferenceResponse,
    SACPatchTSTInferenceRequest,
    SACPatchTSTInferenceResponse,
    SymbolPrediction,
    WeightChange,
)
from .patchtst import router as patchtst_router
from .ppo_lstm import router as ppo_lstm_router
from .ppo_patchtst import router as ppo_patchtst_router
from .sac_lstm import router as sac_lstm_router
from .sac_patchtst import router as sac_patchtst_router

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
    # Request/Response models
    "LSTMInferenceRequest",
    "LSTMInferenceResponse",
    "PPOLSTMInferenceRequest",
    "PPOLSTMInferenceResponse",
    "PPOPatchTSTInferenceRequest",
    "PPOPatchTSTInferenceResponse",
    "PatchTSTInferenceRequest",
    "PatchTSTInferenceResponse",
    "PortfolioSnapshot",
    "Position",
    "SACLSTMInferenceRequest",
    "SACLSTMInferenceResponse",
    "SACPatchTSTInferenceRequest",
    "SACPatchTSTInferenceResponse",
    "SymbolPrediction",
    "WeightChange",
    # Helpers
    "_load_model_artifacts",
    "_load_model_artifacts_generic",
    "_load_patchtst_model_artifacts",
    "_sort_patchtst_predictions",
    "_sort_predictions",
    # Dependencies
    "get_as_of_date",
    "get_patchtst_as_of_date",
    "get_patchtst_storage",
    "get_ppo_lstm_as_of_date",
    "get_ppo_lstm_storage",
    "get_ppo_patchtst_as_of_date",
    "get_ppo_patchtst_storage",
    "get_price_loader",
    "get_sac_lstm_as_of_date",
    "get_sac_lstm_storage",
    "get_sac_patchtst_as_of_date",
    "get_sac_patchtst_storage",
    "get_sentiment_parquet_path",
    "get_storage",
    "get_week_boundary_computer",
    "router",
]

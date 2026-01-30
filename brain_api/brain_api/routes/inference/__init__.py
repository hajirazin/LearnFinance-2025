"""Inference endpoints for ML models.

This module provides inference endpoints for various model types:
- LSTM: Pure price-based weekly return prediction
- PatchTST: Multi-signal weekly return prediction
- PPO: Portfolio allocator using dual forecasts (LSTM + PatchTST)
- SAC: Portfolio allocator using dual forecasts (LSTM + PatchTST)
"""

from fastapi import APIRouter

# Re-export dependencies for backward compatibility
from .dependencies import (
    get_as_of_date,
    get_patchtst_as_of_date,
    get_patchtst_storage,
    get_ppo_as_of_date,
    get_ppo_storage,
    get_price_loader,
    get_sac_as_of_date,
    get_sac_storage,
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
    PPOInferenceRequest,
    PPOInferenceResponse,
    SACInferenceRequest,
    SACInferenceResponse,
    SymbolPrediction,
    WeightChange,
)
from .patchtst import router as patchtst_router
from .ppo import router as ppo_router
from .sac import router as sac_router

# Create combined router
router = APIRouter()

# Include all sub-routers
router.include_router(lstm_router)
router.include_router(patchtst_router)
router.include_router(ppo_router)
router.include_router(sac_router)

__all__ = [
    # Request/Response models
    "LSTMInferenceRequest",
    "LSTMInferenceResponse",
    "PPOInferenceRequest",
    "PPOInferenceResponse",
    "PatchTSTInferenceRequest",
    "PatchTSTInferenceResponse",
    "PortfolioSnapshot",
    "Position",
    "SACInferenceRequest",
    "SACInferenceResponse",
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
    "get_ppo_as_of_date",
    "get_ppo_storage",
    "get_price_loader",
    "get_sac_as_of_date",
    "get_sac_storage",
    "get_sentiment_parquet_path",
    "get_storage",
    "get_week_boundary_computer",
    "router",
]

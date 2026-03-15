"""Training endpoints for ML models.

This module provides training endpoints for various model types:
- LSTM: Pure price-based weekly return prediction
- PatchTST: OHLCV 5-channel weekly return prediction
- SAC: Portfolio allocator using dual forecasts (LSTM + PatchTST)
"""

from fastapi import APIRouter, HTTPException

# Re-export dependencies for backward compatibility
from .dependencies import (
    get_config,
    get_dataset_builder,
    get_forecaster_training_symbols,
    get_patchtst_config,
    get_patchtst_dataset_builder,
    get_patchtst_price_loader,
    get_patchtst_storage,
    get_patchtst_trainer,
    get_price_loader,
    get_rl_training_symbols,
    get_sac_config,
    get_sac_storage,
    get_storage,
    get_top15_symbols,
    get_trainer,
    snapshots_available,
)
from .job_registry import get_job

# Re-export internal functions for backward compatibility
from .lstm import _backfill_lstm_snapshots
from .lstm import router as lstm_router

# Re-export response models for backward compatibility
from .models import (
    LSTMTrainResponse,
    PatchTSTTrainResponse,
    SACTrainResponse,
    TrainingJobStatusResponse,
)
from .patchtst import _backfill_patchtst_snapshots
from .patchtst import router as patchtst_router
from .patchtst_india import router as patchtst_india_router
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
router.include_router(patchtst_india_router)
router.include_router(sac_router)


@router.get("/status/{job_id}", response_model=TrainingJobStatusResponse)
def get_training_job_status(job_id: str) -> TrainingJobStatusResponse:
    """Get the status of a training job.

    Args:
        job_id: The job ID returned from a training POST endpoint.

    Returns:
        Current status, progress, and result (when completed).
    """
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")

    return TrainingJobStatusResponse(
        job_id=job.job_id,
        model_type=job.model_type,
        status=job.status,
        started_at=job.started_at.isoformat(),
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        progress=job.progress,
        error=job.error,
        result=job.result,
    )


__all__ = [
    # Response models
    "LSTMTrainResponse",
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
    "get_patchtst_dataset_builder",
    "get_patchtst_price_loader",
    "get_patchtst_storage",
    "get_patchtst_trainer",
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

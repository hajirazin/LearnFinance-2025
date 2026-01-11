"""SAC + PatchTST inference endpoint."""

import logging
import time

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from brain_api.core.inference_utils import compute_week_from_cutoff
from brain_api.core.sac_patchtst import run_sac_inference as run_sac_patchtst_inference
from brain_api.storage.local import SACPatchTSTLocalStorage

from .dependencies import get_sac_patchtst_as_of_date, get_sac_patchtst_storage
from .models import (
    SACPatchTSTInferenceRequest,
    SACPatchTSTInferenceResponse,
    WeightChange,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/sac_patchtst", response_model=SACPatchTSTInferenceResponse)
def infer_sac_patchtst(
    request: SACPatchTSTInferenceRequest,
    storage: SACPatchTSTLocalStorage = Depends(get_sac_patchtst_storage),
) -> SACPatchTSTInferenceResponse:
    """Get target portfolio weights from SAC + PatchTST policy.

    This endpoint:
    1. Loads the current SAC + PatchTST model
    2. Normalizes the portfolio snapshot to weights
    3. Builds state vector with current signals + PatchTST forecasts
    4. Runs SAC inference to get target weights
    5. Returns target weights and turnover

    Args:
        request: Portfolio snapshot (cash + positions)

    Returns:
        Target weights and execution metadata
    """
    t_start = time.time()
    logger.info("[SAC_PatchTST] Starting inference")

    # Get cutoff date (always a Friday)
    cutoff_date = get_sac_patchtst_as_of_date(request)
    logger.info(f"[SAC_PatchTST] Cutoff date: {cutoff_date}")

    # Compute target week boundaries for the week AFTER cutoff
    week_boundaries = compute_week_from_cutoff(cutoff_date)
    logger.info(
        f"[SAC_PatchTST] Target week: {week_boundaries.target_week_start} to {week_boundaries.target_week_end}"
    )

    # Load model artifacts
    logger.info("[SAC_PatchTST] Loading model artifacts...")
    try:
        artifacts = storage.load_current_artifacts()
    except ValueError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e),
        ) from e

    logger.info(f"[SAC_PatchTST] Model loaded: version={artifacts.version}")

    # Convert portfolio snapshot to values dict
    cash_value = request.portfolio.cash
    position_values = {
        pos.symbol: pos.market_value for pos in request.portfolio.positions
    }

    # Compute total portfolio value
    total_value = cash_value + sum(position_values.values())

    # Build current weights vector (including CASH)
    n_stocks = len(artifacts.symbol_order)
    current_weights = np.zeros(n_stocks + 1)
    for i, symbol in enumerate(artifacts.symbol_order):
        if symbol in position_values and total_value > 0:
            current_weights[i] = position_values[symbol] / total_value
    current_weights[-1] = cash_value / total_value if total_value > 0 else 1.0

    # Build real-time signals (news sentiment + fundamentals)
    logger.info("[SAC_PatchTST] Fetching real-time signals...")
    from .helpers import build_current_forecasts, build_current_signals

    signals = build_current_signals(artifacts.symbol_order, cutoff_date)

    # Build PatchTST forecast features
    logger.info("[SAC_PatchTST] Generating PatchTST forecasts...")
    forecast_features = build_current_forecasts(
        artifacts.symbol_order, forecaster_type="patchtst", as_of_date=cutoff_date
    )

    # Run inference
    logger.info("[SAC_PatchTST] Running inference...")
    result = run_sac_patchtst_inference(
        actor=artifacts.actor,
        scaler=artifacts.scaler,
        config=artifacts.config,
        symbol_order=artifacts.symbol_order,
        current_weights=current_weights,
        signals=signals,
        forecast_features=forecast_features,
        model_version=artifacts.version,
    )

    # Build weight changes list
    weight_changes = []
    for symbol in artifacts.symbol_order:
        current_w = current_weights[artifacts.symbol_order.index(symbol)]
        target_w = result.allocation.get(symbol, 0.0)
        weight_changes.append(
            WeightChange(
                symbol=symbol,
                current_weight=current_w,
                target_weight=target_w,
                change=target_w - current_w,
            )
        )
    # Add CASH
    weight_changes.append(
        WeightChange(
            symbol="CASH",
            current_weight=current_weights[-1],
            target_weight=result.allocation.get("CASH", 0.0),
            change=result.allocation.get("CASH", 0.0) - current_weights[-1],
        )
    )

    t_total = time.time() - t_start
    logger.info(
        f"[SAC_PatchTST] Inference complete in {t_total:.2f}s, turnover={result.turnover:.4f}"
    )

    return SACPatchTSTInferenceResponse(
        target_weights=result.allocation,
        turnover=result.turnover,
        target_week_start=week_boundaries.target_week_start.isoformat(),
        target_week_end=week_boundaries.target_week_end.isoformat(),
        model_version=result.model_version,
        weight_changes=weight_changes,
    )

"""PPO + PatchTST inference endpoint."""

import logging
import time

from fastapi import APIRouter, Depends, HTTPException

from brain_api.core.lstm import compute_week_boundaries
from brain_api.core.ppo_patchtst import run_ppo_patchtst_inference
from brain_api.storage.local import PPOPatchTSTLocalStorage

from .dependencies import get_ppo_patchtst_as_of_date, get_ppo_patchtst_storage
from .models import (
    PPOPatchTSTInferenceRequest,
    PPOPatchTSTInferenceResponse,
    WeightChange,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/ppo_patchtst", response_model=PPOPatchTSTInferenceResponse)
def infer_ppo_patchtst(
    request: PPOPatchTSTInferenceRequest,
    storage: PPOPatchTSTLocalStorage = Depends(get_ppo_patchtst_storage),
) -> PPOPatchTSTInferenceResponse:
    """Get target portfolio weights from PPO + PatchTST policy.

    This endpoint:
    1. Loads the current PPO + PatchTST model
    2. Normalizes the portfolio snapshot to weights
    3. Builds state vector with current signals + PatchTST forecasts
    4. Runs PPO inference to get target weights
    5. Returns target weights and turnover

    Args:
        request: Portfolio snapshot (cash + positions)

    Returns:
        Target weights and execution metadata
    """
    t_start = time.time()
    logger.info("[PPO_PatchTST] Starting inference")

    # Get as-of date
    as_of = get_ppo_patchtst_as_of_date(request)
    logger.info(f"[PPO_PatchTST] As-of date: {as_of}")

    # Compute target week boundaries
    week_boundaries = compute_week_boundaries(as_of)
    logger.info(
        f"[PPO_PatchTST] Target week: {week_boundaries.target_week_start} to {week_boundaries.target_week_end}"
    )

    # Load model artifacts
    logger.info("[PPO_PatchTST] Loading model artifacts...")
    try:
        artifacts = storage.load_current_artifacts()
    except ValueError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e),
        ) from e

    logger.info(f"[PPO_PatchTST] Model loaded: version={artifacts.version}")

    # Convert portfolio snapshot to values dict
    cash_value = request.portfolio.cash
    position_values = {
        pos.symbol: pos.market_value for pos in request.portfolio.positions
    }

    # Build real-time signals (news sentiment + fundamentals)
    logger.info("[PPO_PatchTST] Fetching real-time signals...")
    from .helpers import build_current_forecasts, build_current_signals

    signals = build_current_signals(artifacts.symbol_order, as_of)

    # Build PatchTST forecast features
    logger.info("[PPO_PatchTST] Generating PatchTST forecasts...")
    forecast_features = build_current_forecasts(
        artifacts.symbol_order, forecaster_type="patchtst", as_of_date=as_of
    )

    # Run inference
    logger.info("[PPO_PatchTST] Running inference...")
    result = run_ppo_patchtst_inference(
        model=artifacts.model,
        scaler=artifacts.scaler,
        config=artifacts.config,
        symbol_order=artifacts.symbol_order,
        signals=signals,
        forecast_features=forecast_features,
        cash_value=cash_value,
        position_values=position_values,
        target_week_start=week_boundaries.target_week_start,
        target_week_end=week_boundaries.target_week_end,
        model_version=artifacts.version,
    )

    # Build weight changes list
    weight_changes = []
    for symbol in artifacts.symbol_order:
        weight_changes.append(
            WeightChange(
                symbol=symbol,
                current_weight=result.current_weights.get(symbol, 0.0),
                target_weight=result.target_weights.get(symbol, 0.0),
                change=result.weight_changes.get(symbol, 0.0),
            )
        )
    # Add CASH
    weight_changes.append(
        WeightChange(
            symbol="CASH",
            current_weight=result.current_weights.get("CASH", 0.0),
            target_weight=result.target_weights.get("CASH", 0.0),
            change=result.weight_changes.get("CASH", 0.0),
        )
    )

    t_total = time.time() - t_start
    logger.info(
        f"[PPO_PatchTST] Inference complete in {t_total:.2f}s, turnover={result.turnover:.4f}"
    )

    return PPOPatchTSTInferenceResponse(
        target_weights=result.target_weights,
        turnover=result.turnover,
        target_week_start=result.target_week_start,
        target_week_end=result.target_week_end,
        model_version=result.model_version,
        weight_changes=weight_changes,
    )

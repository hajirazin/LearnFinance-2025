"""PPO inference endpoint with dual forecasts (LSTM + PatchTST)."""

import logging
import time

from fastapi import APIRouter, Depends, HTTPException

from brain_api.core.inference_utils import compute_week_from_cutoff
from brain_api.core.ppo import run_ppo_inference
from brain_api.storage.ppo import PPOLocalStorage

from .dependencies import get_ppo_as_of_date, get_ppo_storage
from .models import PPOInferenceRequest, PPOInferenceResponse, WeightChange

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/ppo", response_model=PPOInferenceResponse)
def infer_ppo(
    request: PPOInferenceRequest,
    storage: PPOLocalStorage = Depends(get_ppo_storage),
) -> PPOInferenceResponse:
    """Get target portfolio weights from PPO policy.

    This endpoint:
    1. Loads the current PPO model
    2. Normalizes the portfolio snapshot to weights
    3. Builds state vector with current signals + dual forecasts (LSTM + PatchTST)
    4. Runs PPO inference to get target weights
    5. Returns target weights and turnover

    Args:
        request: Portfolio snapshot (cash + positions)

    Returns:
        Target weights and execution metadata
    """
    t_start = time.time()
    logger.info("[PPO] Starting inference")

    # Get cutoff date (always a Friday)
    cutoff_date = get_ppo_as_of_date(request)
    logger.info(f"[PPO] Cutoff date: {cutoff_date}")

    # Compute target week boundaries for the week AFTER cutoff
    week_boundaries = compute_week_from_cutoff(cutoff_date)
    logger.info(
        f"[PPO] Target week: {week_boundaries.target_week_start} to {week_boundaries.target_week_end}"
    )

    # Load model artifacts
    logger.info("[PPO] Loading model artifacts...")
    try:
        artifacts = storage.load_current_artifacts()
    except ValueError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e),
        ) from e

    logger.info(f"[PPO] Model loaded: version={artifacts.version}")

    # Convert portfolio snapshot to values dict
    cash_value = request.portfolio.cash
    position_values = {
        pos.symbol: pos.market_value for pos in request.portfolio.positions
    }

    # Build real-time signals (news sentiment + fundamentals)
    logger.info("[PPO] Fetching real-time signals...")
    from .helpers import build_current_forecasts, build_current_signals

    signals = build_current_signals(artifacts.symbol_order, cutoff_date)

    # Build dual forecast features (LSTM + PatchTST)
    logger.info("[PPO] Generating LSTM forecasts...")
    lstm_forecasts = build_current_forecasts(
        artifacts.symbol_order, forecaster_type="lstm", as_of_date=cutoff_date
    )

    logger.info("[PPO] Generating PatchTST forecasts...")
    patchtst_forecasts = build_current_forecasts(
        artifacts.symbol_order, forecaster_type="patchtst", as_of_date=cutoff_date
    )

    # Run inference
    logger.info("[PPO] Running inference...")
    result = run_ppo_inference(
        model=artifacts.model,
        scaler=artifacts.scaler,
        config=artifacts.config,
        symbol_order=artifacts.symbol_order,
        signals=signals,
        lstm_forecasts=lstm_forecasts,
        patchtst_forecasts=patchtst_forecasts,
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
        f"[PPO] Inference complete in {t_total:.2f}s, turnover={result.turnover:.4f}"
    )

    return PPOInferenceResponse(
        target_weights=result.target_weights,
        turnover=result.turnover,
        target_week_start=result.target_week_start,
        target_week_end=result.target_week_end,
        model_version=result.model_version,
        weight_changes=weight_changes,
    )

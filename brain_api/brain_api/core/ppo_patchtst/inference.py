"""PPO + PatchTST inference implementation.

Runs PPO inference to get target portfolio weights given current state.
Uses PatchTST predictions as the forecast feature.
"""

from __future__ import annotations

from datetime import date

from brain_api.core.portfolio_rl.scaler import PortfolioScaler
from brain_api.core.ppo_patchtst.config import PPOPatchTSTConfig
from brain_api.core.ppo_lstm.model import PPOActorCritic
from brain_api.core.ppo_lstm.inference import (
    PPOInferenceResult,
    run_ppo_inference as _run_ppo_inference_base,
)


def run_ppo_patchtst_inference(
    model: PPOActorCritic,
    scaler: PortfolioScaler,
    config: PPOPatchTSTConfig,
    symbol_order: list[str],
    signals: dict[str, dict[str, float]],
    forecast_features: dict[str, float],  # PatchTST predictions
    cash_value: float,
    position_values: dict[str, float],
    target_week_start: date,
    target_week_end: date,
    model_version: str,
) -> PPOInferenceResult:
    """Run PPO inference to get target portfolio weights.
    
    This is functionally identical to run_ppo_inference from ppo_lstm,
    but expects PatchTST forecasts instead of LSTM forecasts.
    
    Args:
        model: Trained PPO model.
        scaler: Fitted state scaler.
        config: PPO configuration.
        symbol_order: Ordered list of symbols.
        signals: Current signals for each symbol.
        forecast_features: PatchTST predictions for each symbol.
        cash_value: Current cash balance.
        position_values: Current position values.
        target_week_start: Start of target week.
        target_week_end: End of target week.
        model_version: Model version string.
    
    Returns:
        PPOInferenceResult with target weights and metadata.
    """
    # Reuse the base implementation - it's agnostic to forecast source
    return _run_ppo_inference_base(
        model=model,
        scaler=scaler,
        config=config,
        symbol_order=symbol_order,
        signals=signals,
        forecast_features=forecast_features,
        cash_value=cash_value,
        position_values=position_values,
        target_week_start=target_week_start,
        target_week_end=target_week_end,
        model_version=model_version,
    )


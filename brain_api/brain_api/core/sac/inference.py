"""SAC inference implementation with dual forecasts (LSTM + PatchTST)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from brain_api.core.portfolio_rl.constraints import (
    apply_softmax_to_weights,
    compute_turnover,
    enforce_constraints,
)
from brain_api.core.portfolio_rl.state import build_state_vector

if TYPE_CHECKING:
    from brain_api.core.portfolio_rl.sac_networks import GaussianActor
    from brain_api.core.portfolio_rl.scaler import PortfolioScaler
    from brain_api.core.sac.config import SACConfig


@dataclass
class SACInferenceResult:
    """Result from SAC inference."""

    allocation: dict[str, float]  # symbol -> weight
    turnover: float
    model_version: str
    raw_action: np.ndarray


def run_sac_inference(
    actor: GaussianActor,
    scaler: PortfolioScaler,
    config: SACConfig,
    symbol_order: list[str],
    current_weights: np.ndarray,
    signals: dict[str, dict[str, float]],
    lstm_forecasts: dict[str, float],
    patchtst_forecasts: dict[str, float],
    model_version: str,
) -> SACInferenceResult:
    """Run SAC inference to get portfolio allocation.

    Args:
        actor: Trained actor network.
        scaler: Fitted state scaler.
        config: SAC configuration.
        symbol_order: Ordered list of symbols.
        current_weights: Current portfolio weights (including CASH).
        signals: Dict of symbol -> dict of signal values.
        lstm_forecasts: Dict of symbol -> LSTM forecast value.
        patchtst_forecasts: Dict of symbol -> PatchTST forecast value.
        model_version: Model version string.

    Returns:
        Inference result with allocation weights.
    """
    # Build state vector with dual forecasts
    state = build_state_vector(
        signals=signals,
        lstm_forecasts=lstm_forecasts,
        patchtst_forecasts=patchtst_forecasts,
        portfolio_weights=current_weights,
        symbol_order=symbol_order,
    )

    # Normalize state
    state_normalized = scaler.transform(state)

    # Get action from actor (deterministic for inference)
    with torch.no_grad():
        action = actor.get_action(state_normalized, deterministic=True)

    # Convert action logits to portfolio weights
    raw_weights = apply_softmax_to_weights(action)

    # Apply constraints
    weights = enforce_constraints(
        raw_weights,
        cash_buffer=config.cash_buffer,
        max_position_weight=config.max_position_weight,
    )

    # Compute turnover
    turnover = compute_turnover(current_weights, weights)

    # Build allocation dict
    allocation = {}
    for i, symbol in enumerate(symbol_order):
        allocation[symbol] = float(weights[i])
    allocation["CASH"] = float(weights[-1])

    return SACInferenceResult(
        allocation=allocation,
        turnover=turnover,
        model_version=model_version,
        raw_action=action,
    )

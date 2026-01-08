"""PPO + LSTM inference implementation.

Runs PPO inference to get target portfolio weights given current state.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import torch

from brain_api.core.portfolio_rl.constraints import (
    apply_softmax_to_weights,
    enforce_constraints,
    compute_turnover,
    normalize_portfolio_from_values,
)
from brain_api.core.portfolio_rl.scaler import PortfolioScaler
from brain_api.core.portfolio_rl.state import build_state_vector, StateSchema
from brain_api.core.ppo_lstm.config import PPOLSTMConfig
from brain_api.core.ppo_lstm.model import PPOActorCritic


@dataclass
class PPOInferenceResult:
    """Result of PPO inference for a single decision point."""
    
    # Target portfolio weights (symbol -> weight)
    target_weights: dict[str, float]
    
    # Expected turnover to reach target
    turnover: float
    
    # Metadata
    target_week_start: str  # ISO date
    target_week_end: str  # ISO date
    model_version: str
    
    # Per-symbol details
    current_weights: dict[str, float]
    weight_changes: dict[str, float]  # target - current
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "target_weights": self.target_weights,
            "turnover": self.turnover,
            "target_week_start": self.target_week_start,
            "target_week_end": self.target_week_end,
            "model_version": self.model_version,
            "current_weights": self.current_weights,
            "weight_changes": self.weight_changes,
        }


def run_ppo_inference(
    model: PPOActorCritic,
    scaler: PortfolioScaler,
    config: PPOLSTMConfig,
    symbol_order: list[str],
    signals: dict[str, dict[str, float]],
    forecast_features: dict[str, float],
    cash_value: float,
    position_values: dict[str, float],
    target_week_start: date,
    target_week_end: date,
    model_version: str,
) -> PPOInferenceResult:
    """Run PPO inference to get target portfolio weights.
    
    Args:
        model: Trained PPO model.
        scaler: Fitted state scaler.
        config: PPO configuration.
        symbol_order: Ordered list of symbols.
        signals: Current signals for each symbol.
        forecast_features: LSTM predictions for each symbol.
        cash_value: Current cash balance.
        position_values: Current position values (symbol -> market value).
        target_week_start: Start of target week.
        target_week_end: End of target week.
        model_version: Model version string.
    
    Returns:
        PPOInferenceResult with target weights and metadata.
    """
    # Normalize current portfolio to weights
    current_weights_array = normalize_portfolio_from_values(
        cash_value=cash_value,
        position_values=position_values,
        symbol_order=symbol_order,
    )
    
    # Build state vector
    state = build_state_vector(
        signals=signals,
        forecast_features=forecast_features,
        portfolio_weights=current_weights_array,
        symbol_order=symbol_order,
    )
    
    # Scale state
    scaled_state = scaler.transform(state)
    
    # Run model inference
    model.eval()
    with torch.no_grad():
        state_t = torch.FloatTensor(scaled_state)
        action, _, _ = model.get_action_and_value(state_t, deterministic=True)
        action_np = action.cpu().numpy().flatten()
    
    # Convert action to weights via softmax
    raw_weights = apply_softmax_to_weights(action_np)
    
    # Enforce constraints
    target_weights_array = enforce_constraints(
        raw_weights,
        cash_buffer=config.cash_buffer,
        max_position_weight=config.max_position_weight,
    )
    
    # Compute turnover
    turnover = compute_turnover(current_weights_array, target_weights_array)
    
    # Build output dicts
    target_weights = {}
    current_weights = {}
    weight_changes = {}
    
    for i, symbol in enumerate(symbol_order):
        target_weights[symbol] = float(target_weights_array[i])
        current_weights[symbol] = float(current_weights_array[i])
        weight_changes[symbol] = float(target_weights_array[i] - current_weights_array[i])
    
    # CASH is last
    target_weights["CASH"] = float(target_weights_array[-1])
    current_weights["CASH"] = float(current_weights_array[-1])
    weight_changes["CASH"] = float(target_weights_array[-1] - current_weights_array[-1])
    
    return PPOInferenceResult(
        target_weights=target_weights,
        turnover=turnover,
        target_week_start=target_week_start.isoformat(),
        target_week_end=target_week_end.isoformat(),
        model_version=model_version,
        current_weights=current_weights,
        weight_changes=weight_changes,
    )


def compute_orders_from_weights(
    target_weights: dict[str, float],
    current_values: dict[str, float],
    cash_value: float,
    prices: dict[str, float],
) -> list[dict[str, Any]]:
    """Compute buy/sell orders to reach target weights.
    
    This is a helper for converting weight changes to actual orders.
    The caller is responsible for executing the orders.
    
    Args:
        target_weights: Target portfolio weights.
        current_values: Current position market values.
        cash_value: Current cash balance.
        prices: Current prices for each symbol.
    
    Returns:
        List of order dicts with symbol, side, qty, notional.
    """
    # Compute total portfolio value
    total_value = cash_value + sum(current_values.values())
    
    orders = []
    
    for symbol, target_weight in target_weights.items():
        if symbol == "CASH":
            continue
        
        current_value = current_values.get(symbol, 0.0)
        target_value = target_weight * total_value
        
        delta = target_value - current_value
        
        if abs(delta) < 1.0:  # Skip tiny changes
            continue
        
        price = prices.get(symbol, 0.0)
        if price <= 0:
            continue
        
        qty = int(abs(delta) / price)  # Round down to whole shares
        if qty == 0:
            continue
        
        orders.append({
            "symbol": symbol,
            "side": "buy" if delta > 0 else "sell",
            "qty": qty,
            "notional": abs(delta),
        })
    
    # Sort: sells first (to free up cash), then buys
    orders.sort(key=lambda o: (0 if o["side"] == "sell" else 1, -o["notional"]))
    
    return orders


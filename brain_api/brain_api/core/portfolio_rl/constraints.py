"""Portfolio constraint enforcement.

Handles:
- Long-only simplex weights (sum to 1, all >= 0)
- Cash buffer (CASH >= 2%)
- Max position size (each stock <= 20%)
"""

from __future__ import annotations

import numpy as np


def apply_softmax_to_weights(logits: np.ndarray) -> np.ndarray:
    """Apply softmax to convert raw logits to portfolio weights.

    This enforces:
    - All weights >= 0
    - Weights sum to 1.0

    Args:
        logits: Raw policy outputs, shape (n_assets,) or (batch, n_assets)
                Last dimension is CASH.

    Returns:
        Weights on the simplex, same shape as input.
    """
    # Numerical stability: subtract max
    if logits.ndim == 1:
        shifted = logits - np.max(logits)
        exp_logits = np.exp(shifted)
        return exp_logits / np.sum(exp_logits)
    else:
        # Batch case
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(shifted)
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def enforce_constraints(
    weights: np.ndarray,
    cash_buffer: float = 0.02,
    max_position_weight: float = 0.20,
) -> np.ndarray:
    """Enforce portfolio constraints via post-processing.

    Constraints:
    1. All weights >= 0 (already guaranteed by softmax)
    2. Weights sum to 1.0 (already guaranteed by softmax)
    3. Cash weight >= cash_buffer
    4. Each stock weight <= max_position_weight

    The enforcement is done via clipping and renormalization.

    Args:
        weights: Portfolio weights with CASH as last element.
                 Shape (n_assets,) where n_assets = n_stocks + 1.
        cash_buffer: Minimum cash weight (default 0.02 = 2%).
        max_position_weight: Maximum weight per stock (default 0.20 = 20%).

    Returns:
        Constrained weights that sum to 1.0.
    """
    weights = weights.copy()
    n_assets = len(weights)
    cash_idx = n_assets - 1  # CASH is last

    # Step 1: Clip stock weights to max_position_weight
    for i in range(cash_idx):
        if weights[i] > max_position_weight:
            excess = weights[i] - max_position_weight
            weights[i] = max_position_weight
            # Add excess to cash
            weights[cash_idx] += excess

    # Step 2: Ensure cash buffer
    if weights[cash_idx] < cash_buffer:
        deficit = cash_buffer - weights[cash_idx]
        weights[cash_idx] = cash_buffer

        # Reduce stock weights proportionally to cover deficit
        stock_weights = weights[:cash_idx]
        stock_total = np.sum(stock_weights)

        if stock_total > 0:
            reduction_factor = (stock_total - deficit) / stock_total
            reduction_factor = max(0, reduction_factor)  # Don't go negative
            weights[:cash_idx] = stock_weights * reduction_factor

    # Step 3: Renormalize to ensure sum = 1.0 (handle numerical drift)
    total = np.sum(weights)
    if total > 0:
        weights = weights / total
    else:
        # Edge case: all weights are 0, put everything in cash
        weights = np.zeros(n_assets)
        weights[cash_idx] = 1.0

    # Final safety check: ensure constraints are met
    weights = np.clip(weights, 0.0, 1.0)

    # Ensure cash buffer one more time after renormalization
    if weights[cash_idx] < cash_buffer:
        weights[cash_idx] = cash_buffer
        remaining = 1.0 - cash_buffer
        stock_sum = np.sum(weights[:cash_idx])
        if stock_sum > 0:
            weights[:cash_idx] = weights[:cash_idx] * (remaining / stock_sum)

    return weights


def compute_turnover(
    current_weights: np.ndarray,
    target_weights: np.ndarray,
) -> float:
    """Compute portfolio turnover between current and target weights.

    Turnover = 0.5 * sum(|w_target - w_current|)

    This gives turnover in [0, 1] where:
    - 0 = no change
    - 1 = complete portfolio flip (sell everything, buy new)

    Args:
        current_weights: Current portfolio weights (n_assets,)
        target_weights: Target portfolio weights (n_assets,)

    Returns:
        Turnover as a decimal (0 to 1).
    """
    return 0.5 * np.sum(np.abs(target_weights - current_weights))


def normalize_portfolio_from_values(
    cash_value: float,
    position_values: dict[str, float],
    symbol_order: list[str],
) -> np.ndarray:
    """Normalize raw portfolio values into weights.

    Args:
        cash_value: Cash balance in dollars.
        position_values: Dict of symbol -> market value in dollars.
        symbol_order: Ordered list of symbols (must match action space order).

    Returns:
        Weights array with stocks first, CASH last.
    """
    n_assets = len(symbol_order) + 1  # +1 for CASH

    # Compute total portfolio value
    total_value = cash_value + sum(position_values.values())

    if total_value <= 0:
        # Edge case: empty portfolio, return all cash
        weights = np.zeros(n_assets)
        weights[-1] = 1.0  # CASH is last
        return weights

    # Build weights array
    weights = np.zeros(n_assets)
    for i, symbol in enumerate(symbol_order):
        if symbol in position_values:
            weights[i] = position_values[symbol] / total_value

    # CASH is last
    weights[-1] = cash_value / total_value

    return weights


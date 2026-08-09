"""Portfolio constraint enforcement.

Handles:
- Long-only simplex weights (sum to 1, all >= 0) via masked softmax
- Cash buffer (CASH >= cash_buffer, default 2%)

Position-size caps are not enforced here; concentration is shaped by the
masked tanh actor, entropy target ``-(n_valid + 1)``, and the HHI penalty
in the reward.
"""

from __future__ import annotations

import numpy as np


def apply_softmax_to_weights(
    logits: np.ndarray, asset_mask: np.ndarray | None = None
) -> np.ndarray:
    """Apply masked softmax to convert raw logits to portfolio weights.

    This enforces:
    - All weights >= 0
    - Weights sum to 1.0
    - Masked/padded stock slots receive exactly weight 0
    - CASH is always a valid destination

    Args:
        logits: Raw policy outputs, shape (n_assets,) or (batch, n_assets)
                Last dimension is CASH. Actor outputs are tanh-bounded to
                ``[-1, 1]``; warmup paths should match that bound.
        asset_mask: Optional boolean mask over stock slots only (excluding
                CASH). Shape ``(n_stocks,)`` or ``(batch, n_stocks)``.

    Returns:
        Weights on the simplex, same shape as input.
    """
    values = np.asarray(logits, dtype=np.float64)
    single = values.ndim == 1
    batch = values.reshape(1, -1) if single else values
    if batch.ndim != 2:
        raise ValueError("logits must be one- or two-dimensional")
    if asset_mask is None:
        action_mask = np.ones_like(batch, dtype=bool)
    else:
        mask = np.asarray(asset_mask, dtype=bool)
        if mask.ndim == 1:
            mask = mask.reshape(1, -1)
        if mask.shape[0] == 1 and batch.shape[0] > 1:
            mask = np.repeat(mask, batch.shape[0], axis=0)
        if mask.shape != (batch.shape[0], batch.shape[1] - 1):
            raise ValueError("asset_mask must match logits excluding CASH")
        action_mask = np.concatenate(
            (mask, np.ones((batch.shape[0], 1), dtype=bool)), axis=1
        )
    masked = np.where(action_mask, batch, -np.inf)
    shifted = masked - np.max(masked, axis=-1, keepdims=True)
    exp_logits = np.where(action_mask, np.exp(shifted), 0.0)
    weights = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
    weights[~action_mask] = 0.0
    return weights[0] if single else weights


def enforce_constraints(
    weights: np.ndarray,
    cash_buffer: float = 0.02,
) -> np.ndarray:
    """Enforce the cash-buffer constraint via clipping and renormalization.

    Constraints:
    1. All weights >= 0 (already guaranteed by masked softmax)
    2. Weights sum to 1.0 (already guaranteed by masked softmax)
    3. Cash weight >= cash_buffer

    Masked stock zeros stay zero because stock mass is only rescaled
    proportionally. There is no per-name max-weight clip here.

    Args:
        weights: Portfolio weights with CASH as last element.
                 Shape ``(ACTION_DIM,)`` = 30 stock slots + CASH, or a
                 shorter ``(n_stocks + 1,)`` vector in legacy helpers.
        cash_buffer: Minimum cash weight (default 0.02 = 2%).

    Returns:
        Constrained weights that sum to 1.0.
    """
    weights = weights.copy()
    n_assets = len(weights)
    cash_idx = n_assets - 1  # CASH is last

    # Step 1: Ensure cash buffer
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

    # Step 2: Renormalize to ensure sum = 1.0 (handle numerical drift)
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
    current = np.asarray(current_weights, dtype=np.float64)
    target = np.asarray(target_weights, dtype=np.float64)
    if current.shape != target.shape:
        raise ValueError("current_weights and target_weights must share a shape")
    return float(0.5 * np.sum(np.abs(target - current)))


def compute_turnover_from_allocations(
    current_allocation: dict[str, float],
    target_allocation: dict[str, float],
) -> float:
    """Turnover over the union of named sleeves (including forced liquidations).

    Off-slate names that must be sold to weight 0 participate explicitly, so
    reported turnover matches broker churn rather than a cash-folded view.
    """
    symbols = sorted(set(current_allocation) | set(target_allocation))
    current = np.asarray(
        [float(current_allocation.get(symbol, 0.0)) for symbol in symbols],
        dtype=np.float64,
    )
    target = np.asarray(
        [float(target_allocation.get(symbol, 0.0)) for symbol in symbols],
        dtype=np.float64,
    )
    return compute_turnover(current, target)


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

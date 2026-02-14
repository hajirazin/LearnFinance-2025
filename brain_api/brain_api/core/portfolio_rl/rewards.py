"""Reward computation for portfolio RL.

Reward = scaled(portfolio_log_return - log(1 + transaction_cost))

Both terms are in log space for mathematical consistency.
All rewards are scaled by reward_scale (default 100) so that
a 1% weekly return becomes a reward of 1.0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from brain_api.core.portfolio_rl.config import PPOBaseConfig


def compute_portfolio_return(
    weights: np.ndarray,
    symbol_returns: np.ndarray,
) -> float:
    """Compute portfolio return from weights and asset returns.

    Args:
        weights: Portfolio weights (n_assets,) with CASH as last element.
                 Weights should sum to 1.0.
        symbol_returns: Weekly returns for each asset (n_assets,).
                       CASH return is typically 0 (or risk-free rate).

    Returns:
        Portfolio return as a decimal (e.g., 0.02 for 2%).
    """
    return float(np.dot(weights, symbol_returns))


def compute_portfolio_log_return(
    weights: np.ndarray,
    symbol_returns: np.ndarray,
) -> float:
    """Compute portfolio log return.

    For small returns, log(1 + r) ≈ r, so this is approximately
    equal to simple return. Log returns are additive across time.

    Args:
        weights: Portfolio weights (n_assets,) with CASH as last element.
        symbol_returns: Weekly returns for each asset (n_assets,).

    Returns:
        Portfolio log return.
    """
    simple_return = compute_portfolio_return(weights, symbol_returns)
    # Clamp to avoid log(0) or log(negative)
    return float(np.log(max(1 + simple_return, 1e-10)))


def compute_transaction_cost(
    turnover: float,
    cost_bps: int = 10,
) -> float:
    """Compute transaction cost from turnover.

    Args:
        turnover: Portfolio turnover (0 to 1).
        cost_bps: Cost in basis points per unit turnover (default 10).

    Returns:
        Transaction cost as a decimal (e.g., 0.001 for 0.1%).
    """
    cost_rate = cost_bps / 10_000
    return turnover * cost_rate


def compute_reward(
    portfolio_return: float,
    turnover: float,
    config: PPOBaseConfig,
) -> float:
    """Compute scaled reward for RL training.

    Reward = reward_scale * (portfolio_return - transaction_cost)

    With default reward_scale=100:
    - 1% weekly return with no turnover → reward = 1.0
    - 1% weekly return with 50% turnover at 10bps → reward = 1.0 - 0.05 = 0.95

    Args:
        portfolio_return: Simple portfolio return (decimal).
        turnover: Portfolio turnover (0 to 1).
        config: PPO config with cost_bps and reward_scale.

    Returns:
        Scaled reward for RL training.
    """
    transaction_cost = compute_transaction_cost(turnover, config.cost_bps)
    net_return = portfolio_return - transaction_cost
    return net_return * config.reward_scale


def compute_reward_from_log_return(
    portfolio_log_return: float,
    turnover: float,
    config: PPOBaseConfig,
) -> float:
    """Compute scaled reward using log return.

    Both the portfolio return and transaction cost are in log space:
      net_return = log(1 + r) - log(1 + tc) = log((1 + r) / (1 + tc))

    This ensures mathematical consistency -- subtracting a linear cost
    from a log return would mix units.

    Args:
        portfolio_log_return: Log portfolio return, i.e. log(1 + r).
        turnover: Portfolio turnover (0 to 1).
        config: PPO config.

    Returns:
        Scaled reward for RL training.
    """
    transaction_cost = compute_transaction_cost(turnover, config.cost_bps)
    net_return = portfolio_log_return - np.log(1 + transaction_cost)
    return net_return * config.reward_scale

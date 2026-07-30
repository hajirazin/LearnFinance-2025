"""Reward computation for portfolio RL.

Reward = scaled(log(1 + gross_return - transaction_cost_fraction)).

Transaction costs are deducted from wealth before the logarithm is taken.
All rewards are scaled by reward_scale (default 100) so that
a 1% weekly return becomes a reward of 1.0.

Includes Differential Sharpe Ratio (Moody & Saffell 2001) for
risk-adjusted reward shaping.

Cost source
-----------
The ``transaction_cost_fraction`` argument is **always pre-computed
by the caller**. The canonical implementation lives in
:mod:`brain_api.core.portfolio_rl.broker_costs` (IBKR Singapore
Tiered: per-symbol commission with min/max, sell-side regulatory,
clearing, pass-through). This module owns the *shape* of the reward
formula; the broker cost model owns the *amount*.

The legacy flat ``cost_bps * turnover`` formula
(:func:`compute_transaction_cost`) is retained as a deprecation
shim so any existing callers / experience records that still pass
``turnover`` continue to work for one cycle. New code paths must use
:mod:`broker_costs` and pass the resulting fraction in directly.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from brain_api.core.portfolio_rl.config import RLBaseConfig


@dataclass(frozen=True)
class RebalanceTransition:
    """Exact after-cost wealth and post-return portfolio accounting."""

    gross_return: float
    cost_fraction: float
    net_growth: float
    net_log_return: float
    post_weights: np.ndarray

    @classmethod
    def calculate(
        cls,
        target_weights: np.ndarray,
        stock_returns: np.ndarray,
        cost_fraction: float,
    ) -> RebalanceTransition:
        """Calculate one rebalance-to-next-open transition without approximations."""
        target_weights = np.asarray(target_weights, dtype=float)
        stock_returns = np.asarray(stock_returns, dtype=float)
        if target_weights.shape != (len(stock_returns) + 1,):
            raise ValueError(
                "target_weights must contain one weight per stock plus CASH"
            )
        if not np.all(np.isfinite(target_weights)):
            raise ValueError("target_weights must all be finite")
        if np.any(target_weights < 0) or not np.isclose(
            float(target_weights.sum()), 1.0, atol=1e-8
        ):
            raise ValueError("target_weights must be a nonnegative simplex")
        if not np.all(np.isfinite(stock_returns)):
            raise ValueError("stock_returns must be complete and finite")
        if not np.isfinite(cost_fraction) or cost_fraction < 0:
            raise ValueError("cost_fraction must be finite and nonnegative")

        gross_return = float(np.dot(target_weights[:-1], stock_returns))
        net_growth = 1.0 + gross_return - cost_fraction
        if not np.isfinite(net_growth) or net_growth <= 0:
            raise ValueError(
                f"Rebalance transition has invalid net growth {net_growth}"
            )

        post_weights = np.empty_like(target_weights)
        post_weights[:-1] = target_weights[:-1] * (1.0 + stock_returns) / net_growth
        post_weights[-1] = (target_weights[-1] - cost_fraction) / net_growth
        if not np.all(np.isfinite(post_weights)) or np.any(post_weights < -1e-12):
            raise ValueError(
                "Rebalance transition produced invalid post-return weights"
            )
        post_weights = np.maximum(post_weights, 0.0)
        if not np.isclose(float(post_weights.sum()), 1.0, atol=1e-8):
            raise ValueError(
                f"Post-return weights must sum to 1.0, got {post_weights.sum()}"
            )

        return cls(
            gross_return=gross_return,
            cost_fraction=float(cost_fraction),
            net_growth=net_growth,
            net_log_return=float(np.log(net_growth)),
            post_weights=post_weights,
        )


def compute_net_log_reward(
    gross_return: float,
    transaction_cost_fraction: float,
    config: RLBaseConfig,
    target_weights: np.ndarray | None = None,
) -> float:
    """Compute exact net-log-wealth reward plus the existing HHI penalty."""
    reward = (
        compute_exact_net_log_return(gross_return, transaction_cost_fraction)
        * config.reward_scale
    )

    if target_weights is not None and getattr(config, "hhi_penalty_scale", 0.0) > 0:
        stock_weights = target_weights[: config.n_stocks]
        hhi = float(np.sum(stock_weights**2))
        if hhi > 0.20:
            reward -= (hhi - 0.20) * config.hhi_penalty_scale * config.reward_scale
    return reward


def compute_exact_net_log_return(
    gross_return: float, transaction_cost_fraction: float
) -> float:
    """Return ``log(1 + gross_return - cost_fraction)`` exactly."""
    if not np.isfinite(gross_return):
        raise ValueError(f"Gross return must be finite, got {gross_return}")
    if not np.isfinite(transaction_cost_fraction) or transaction_cost_fraction < 0:
        raise ValueError(
            "Transaction cost fraction must be finite and nonnegative, "
            f"got {transaction_cost_fraction}"
        )
    net_growth = 1.0 + gross_return - transaction_cost_fraction
    if not np.isfinite(net_growth) or net_growth <= 0:
        raise ValueError(f"Net growth must be finite and positive, got {net_growth}")
    return float(np.log(net_growth))


class DifferentialSharpe:
    """Online incremental Sharpe ratio estimator (Moody & Saffell 2001).

    Computes the differential Sharpe ratio as an incremental reward signal.
    Uses exponential moving averages of returns and squared returns.

    The reward at each step measures how much the current return improves
    the running Sharpe ratio -- positive for returns above risk-adjusted
    expectations, negative for returns that increase risk without return.
    """

    def __init__(self, eta: float = 0.01):
        """Initialize with learning rate eta.

        Args:
            eta: EMA decay rate. Lower = more stable, higher = more responsive.
                 0.01 is standard for weekly data (~100-week effective window).
        """
        self.eta = eta
        self.A = 0.0  # EMA of returns
        self.B = 0.0  # EMA of squared returns

    def update(self, r: float) -> float:
        """Compute differential Sharpe ratio for this step's return.

        Args:
            r: Portfolio return for this step (simple return, not log).

        Returns:
            Differential Sharpe ratio reward.
        """
        dA = r - self.A
        dB = r**2 - self.B
        denominator = (self.B - self.A**2) ** 1.5

        if abs(denominator) < 1e-12:
            # Not enough variance yet (early episodes), return 0
            dsr = 0.0
        else:
            dsr = (self.B * dA - 0.5 * self.A * dB) / denominator

        # Update EMAs
        self.A += self.eta * dA
        self.B += self.eta * dB

        return dsr


def compute_blended_reward(
    portfolio_log_return: float,
    portfolio_simple_return: float,
    transaction_cost_fraction: float,
    differential_sharpe: DifferentialSharpe,
    config: RLBaseConfig,
    target_weights: np.ndarray | None = None,
) -> float:
    """Compute blended reward: return + differential Sharpe.

    reward = sharpe_weight * DSR + (1 - sharpe_weight) * return_reward

    The return component incentivizes making money.
    The DSR component penalizes volatile strategies and rewards
    consistent risk-adjusted performance.

    Args:
        portfolio_log_return: Log portfolio return log(1 + r).
        portfolio_simple_return: Simple portfolio return r.
        transaction_cost_fraction: Pre-computed transaction cost as a
            fraction of NAV (``total_dollar_cost / nav_usd``). The
            canonical source is
            :func:`brain_api.core.portfolio_rl.broker_costs.compute_ibkr_rebalance_cost`
            (IBKR Singapore Tiered model). Must be >= 0; pass 0.0 for
            cost-free episodes (e.g. initial reset).
        differential_sharpe: DifferentialSharpe instance (stateful, updates EMAs).
        config: Config with reward_scale + sharpe_weight (cost_bps no longer read).
        target_weights: Optional weights array (n_stocks + 1) to apply HHI penalty.

    Returns:
        Blended reward for RL training.
    """
    if transaction_cost_fraction < 0:
        raise ValueError(
            f"transaction_cost_fraction must be >= 0, got {transaction_cost_fraction}"
        )

    # Kept as a compatibility wrapper for old callers. DSR is intentionally
    # ignored: Sharpe belongs in evaluation/reporting, not the training reward.
    del portfolio_log_return, differential_sharpe
    return compute_net_log_reward(
        portfolio_simple_return,
        transaction_cost_fraction,
        config,
        target_weights,
    )


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
    """**Deprecated**: legacy flat ``turnover * cost_bps`` cost formula.

    This is the pre-IBKR-SG cost model. New code must compute the
    transaction-cost fraction via
    :func:`brain_api.core.portfolio_rl.broker_costs.compute_ibkr_rebalance_cost`
    and pass the resulting fraction directly to
    :func:`compute_blended_reward` /
    :func:`compute_reward_from_log_return`.

    Retained as a deprecation shim so any in-flight code (or
    experience-buffer records that still carry only ``turnover``)
    keeps producing a number rather than crashing while we migrate.

    Args:
        turnover: Portfolio turnover (0 to 1).
        cost_bps: Cost in basis points per unit turnover (default 10).

    Returns:
        Transaction cost as a decimal (e.g., 0.001 for 0.1%).
    """
    warnings.warn(
        "compute_transaction_cost(turnover, cost_bps) is deprecated; use "
        "broker_costs.compute_ibkr_rebalance_cost(...) and pass the "
        "resulting total_fraction to compute_blended_reward / "
        "compute_reward_from_log_return instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    cost_rate = cost_bps / 10_000
    return turnover * cost_rate


def compute_reward(
    portfolio_return: float,
    transaction_cost_fraction: float,
    config: RLBaseConfig,
) -> float:
    """Compute scaled reward for RL training (simple-return form).

    Reward = reward_scale * (portfolio_return - transaction_cost_fraction)

    Args:
        portfolio_return: Simple portfolio return (decimal).
        transaction_cost_fraction: Pre-computed transaction cost as a
            fraction of NAV (see :func:`compute_blended_reward`).
        config: RL config with reward_scale.

    Returns:
        Scaled reward for RL training.
    """
    if transaction_cost_fraction < 0:
        raise ValueError(
            f"transaction_cost_fraction must be >= 0, got {transaction_cost_fraction}"
        )
    net_return = portfolio_return - transaction_cost_fraction
    return net_return * config.reward_scale


def compute_reward_from_log_return(
    portfolio_log_return: float,
    transaction_cost_fraction: float,
    config: RLBaseConfig,
) -> float:
    """Compute scaled reward using log return.

    Converts the gross log return back to its simple-return equivalent,
    deducts cost from wealth, then takes the exact net logarithm:
    ``log(1 + expm1(portfolio_log_return) - cost_fraction)``.

    Args:
        portfolio_log_return: Log portfolio return, i.e. log(1 + r).
        transaction_cost_fraction: Pre-computed transaction cost as a
            fraction of NAV (see :func:`compute_blended_reward`).
        config: RL config.

    Returns:
        Scaled reward for RL training.
    """
    if transaction_cost_fraction < 0:
        raise ValueError(
            f"transaction_cost_fraction must be >= 0, got {transaction_cost_fraction}"
        )
    gross_return = float(np.expm1(portfolio_log_return))
    return compute_net_log_reward(gross_return, transaction_cost_fraction, config)

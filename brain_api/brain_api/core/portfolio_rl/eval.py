"""Evaluation helpers for portfolio RL.

Computes:
- Sharpe ratio (annualized, after costs)
- CAGR (compound annual growth rate)
- Max drawdown
- Hit rate (% weeks with positive return)
- Turnover statistics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class EvaluationMetrics:
    """Portfolio evaluation metrics."""

    sharpe_ratio: float  # annualized, after costs
    cagr: float  # compound annual growth rate
    max_drawdown: float  # maximum peak-to-trough decline
    hit_rate: float  # % of periods with positive return
    avg_turnover: float  # average weekly turnover
    total_return: float  # cumulative return over period
    n_periods: int  # number of evaluation periods

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sharpe_ratio": self.sharpe_ratio,
            "cagr": self.cagr,
            "max_drawdown": self.max_drawdown,
            "hit_rate": self.hit_rate,
            "avg_turnover": self.avg_turnover,
            "total_return": self.total_return,
            "n_periods": self.n_periods,
        }

    def __str__(self) -> str:
        """Pretty print metrics."""
        return (
            f"Sharpe: {self.sharpe_ratio:.3f} | "
            f"CAGR: {self.cagr*100:.2f}% | "
            f"MaxDD: {self.max_drawdown*100:.2f}% | "
            f"HitRate: {self.hit_rate*100:.1f}% | "
            f"AvgTurnover: {self.avg_turnover*100:.1f}%"
        )


def compute_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 52,
) -> float:
    """Compute annualized Sharpe ratio.

    Sharpe = sqrt(periods_per_year) * (mean_return - rf) / std_return

    Args:
        returns: Array of period returns (e.g., weekly returns).
        risk_free_rate: Risk-free rate per period (default 0).
        periods_per_year: Number of periods per year (52 for weekly).

    Returns:
        Annualized Sharpe ratio.
    """
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - risk_free_rate
    mean_return = np.mean(excess_returns)
    std_return = np.std(excess_returns, ddof=1)

    if std_return < 1e-10:
        return 0.0

    return float(np.sqrt(periods_per_year) * mean_return / std_return)


def compute_cagr(
    returns: np.ndarray,
    periods_per_year: int = 52,
) -> float:
    """Compute compound annual growth rate.

    CAGR = (final_value / initial_value)^(1/years) - 1

    Args:
        returns: Array of period returns.
        periods_per_year: Number of periods per year.

    Returns:
        CAGR as a decimal.
    """
    if len(returns) == 0:
        return 0.0

    # Compute cumulative return
    cumulative = np.prod(1 + returns)

    # Number of years
    n_years = len(returns) / periods_per_year

    if n_years < 1e-10:
        return 0.0

    # CAGR
    if cumulative <= 0:
        return -1.0  # Total loss

    return float(cumulative ** (1 / n_years) - 1)


def compute_max_drawdown(returns: np.ndarray) -> float:
    """Compute maximum drawdown.

    Max drawdown = maximum peak-to-trough decline

    Args:
        returns: Array of period returns.

    Returns:
        Max drawdown as a positive decimal (e.g., 0.20 for 20% drawdown).
    """
    if len(returns) == 0:
        return 0.0

    # Compute cumulative returns (wealth curve)
    cumulative = np.cumprod(1 + returns)

    # Running maximum
    running_max = np.maximum.accumulate(cumulative)

    # Drawdowns
    drawdowns = (running_max - cumulative) / running_max

    return float(np.max(drawdowns))


def compute_hit_rate(returns: np.ndarray) -> float:
    """Compute hit rate (fraction of positive returns).

    Args:
        returns: Array of period returns.

    Returns:
        Hit rate as a decimal (e.g., 0.55 for 55%).
    """
    if len(returns) == 0:
        return 0.0

    return float(np.mean(returns > 0))


def evaluate_policy(
    returns: np.ndarray,
    turnovers: np.ndarray,
    periods_per_year: int = 52,
    risk_free_rate: float = 0.0,
) -> EvaluationMetrics:
    """Compute all evaluation metrics for a policy.

    Args:
        returns: Array of period returns (after costs).
        turnovers: Array of turnover values per period.
        periods_per_year: Number of periods per year.
        risk_free_rate: Risk-free rate per period.

    Returns:
        EvaluationMetrics with all computed metrics.
    """
    return EvaluationMetrics(
        sharpe_ratio=compute_sharpe_ratio(returns, risk_free_rate, periods_per_year),
        cagr=compute_cagr(returns, periods_per_year),
        max_drawdown=compute_max_drawdown(returns),
        hit_rate=compute_hit_rate(returns),
        avg_turnover=float(np.mean(turnovers)) if len(turnovers) > 0 else 0.0,
        total_return=float(np.prod(1 + returns) - 1) if len(returns) > 0 else 0.0,
        n_periods=len(returns),
    )


def compare_policies(
    policy_metrics: EvaluationMetrics,
    baseline_metrics: EvaluationMetrics,
    min_sharpe_improvement: float = 0.0,
) -> bool:
    """Compare policy against baseline.

    Policy is considered better if:
    1. Sharpe ratio is higher by at least min_sharpe_improvement

    Args:
        policy_metrics: Metrics for the policy being evaluated.
        baseline_metrics: Metrics for the baseline.
        min_sharpe_improvement: Minimum improvement in Sharpe required.

    Returns:
        True if policy beats baseline.
    """
    return (
        policy_metrics.sharpe_ratio
        >= baseline_metrics.sharpe_ratio + min_sharpe_improvement
    )


def compute_equal_weight_returns(
    symbol_returns: np.ndarray,
    n_stocks: int,
) -> np.ndarray:
    """Compute returns for equal-weight baseline.

    Args:
        symbol_returns: Array of shape (n_periods, n_stocks).
        n_stocks: Number of stocks.

    Returns:
        Array of portfolio returns for equal-weight strategy.
    """
    # Equal weight across stocks (no cash)
    weights = np.ones(n_stocks) / n_stocks
    return np.dot(symbol_returns, weights)


def compute_baseline_metrics(
    symbol_returns: np.ndarray,
    baseline_type: str = "equal_weight",
    periods_per_year: int = 52,
) -> EvaluationMetrics:
    """Compute metrics for a baseline strategy.

    Args:
        symbol_returns: Array of shape (n_periods, n_stocks).
        baseline_type: Type of baseline ("equal_weight" supported).
        periods_per_year: Periods per year.

    Returns:
        EvaluationMetrics for the baseline.
    """
    n_periods, n_stocks = symbol_returns.shape

    if baseline_type == "equal_weight":
        returns = compute_equal_weight_returns(symbol_returns, n_stocks)
        # Equal weight has zero turnover after initial setup
        turnovers = np.zeros(n_periods)
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")

    return evaluate_policy(returns, turnovers, periods_per_year)


def compute_hrp_baseline_metrics(
    symbol_returns: np.ndarray,
    covariance_lookback: int = 52,
    periods_per_year: int = 52,
) -> EvaluationMetrics:
    """Compute metrics for HRP baseline strategy.

    HRP rebalances weights based on historical covariance.
    For simplicity, we recompute weights once using the first lookback
    period and hold constant (like production).

    Args:
        symbol_returns: Array of shape (n_periods, n_stocks).
        covariance_lookback: Lookback periods for covariance estimation.
        periods_per_year: Periods per year.

    Returns:
        EvaluationMetrics for HRP baseline.
    """
    n_periods, _n_stocks = symbol_returns.shape

    # Use first lookback period to estimate covariance
    lookback_end = min(covariance_lookback, n_periods // 2)
    cov_returns = symbol_returns[:lookback_end]

    if len(cov_returns) < 2:
        # Not enough data, fall back to equal weight
        return compute_baseline_metrics(symbol_returns, "equal_weight", periods_per_year)

    # Compute covariance
    cov = np.cov(cov_returns.T)

    # HRP weights (simplified: inverse variance for diagonal)
    # Full HRP would use hierarchical clustering
    variances = np.diag(cov)
    variances = np.maximum(variances, 1e-10)  # Avoid division by zero
    inv_var = 1.0 / variances
    hrp_weights = inv_var / inv_var.sum()

    # Compute returns using HRP weights
    returns = np.dot(symbol_returns, hrp_weights)

    # HRP has zero turnover (we use static weights)
    turnovers = np.zeros(n_periods)

    return evaluate_policy(returns, turnovers, periods_per_year)


def evaluate_ppo_for_promotion(
    policy_metrics: EvaluationMetrics,
    symbol_returns: np.ndarray,
    prior_sharpe: float | None = None,
    is_first_model: bool = False,
) -> tuple[bool, dict[str, Any]]:
    """Evaluate PPO policy for promotion.

    Promotion rules:
    1. First model is always promoted (so inference has something)
    2. Subsequent models must beat:
       - Equal-weight baseline
       - HRP baseline
       - Prior PPO model (if exists)

    Primary metric: Sharpe ratio (after costs)

    Args:
        policy_metrics: Metrics for the candidate policy.
        symbol_returns: Historical returns for computing baselines.
        prior_sharpe: Prior model's Sharpe ratio (None if first model).
        is_first_model: Whether this is the first model ever.

    Returns:
        Tuple of (should_promote, comparison_details).
    """
    # First model always promoted
    if is_first_model:
        return True, {
            "reason": "first_model",
            "policy_sharpe": policy_metrics.sharpe_ratio,
        }

    # Compute baseline metrics
    eq_weight_metrics = compute_baseline_metrics(symbol_returns, "equal_weight")
    hrp_metrics = compute_hrp_baseline_metrics(symbol_returns)

    comparison = {
        "policy_sharpe": policy_metrics.sharpe_ratio,
        "equal_weight_sharpe": eq_weight_metrics.sharpe_ratio,
        "hrp_sharpe": hrp_metrics.sharpe_ratio,
        "prior_sharpe": prior_sharpe,
        "beats_equal_weight": policy_metrics.sharpe_ratio > eq_weight_metrics.sharpe_ratio,
        "beats_hrp": policy_metrics.sharpe_ratio > hrp_metrics.sharpe_ratio,
        "beats_prior": prior_sharpe is None or policy_metrics.sharpe_ratio > prior_sharpe,
    }

    # Must beat all baselines
    should_promote = (
        comparison["beats_equal_weight"]
        and comparison["beats_hrp"]
        and comparison["beats_prior"]
    )

    comparison["reason"] = "passed_all_gates" if should_promote else "failed_gates"

    return should_promote, comparison


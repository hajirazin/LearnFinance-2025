"""Pure reward-ranking, action-feasibility, and paired-bootstrap calculations."""

import math

import numpy as np
from scipy.optimize import linprog


def bounded_oracle_return(stock_returns, cash_floor=0.02):
    """Gross hindsight ceiling under the actor's ratio bound and cash floor."""
    returns = np.r_[np.asarray(stock_returns, dtype=float), 0.0]
    n = len(returns)
    inequalities = []
    for left in range(n):
        for right in range(n):
            if left != right:
                row = np.zeros(n)
                row[left], row[right] = 1.0, -math.exp(2)
                inequalities.append(row)
    result = linprog(
        -returns,
        A_ub=inequalities,
        b_ub=np.zeros(len(inequalities)),
        A_eq=np.ones((1, n)),
        b_eq=[1],
        bounds=[(0, None)] * (n - 1) + [(cash_floor, None)],
        method="highs",
    )
    if not result.success:
        raise ValueError(f"Oracle linear program failed: {result.message}")
    return float(-result.fun)


def reward_gap(net_return_a, net_return_b, benchmark_return):
    hurdle = math.log1p(benchmark_return)
    return (math.log1p(net_return_a) - hurdle) - (math.log1p(net_return_b) - hurdle)


def action_bounds(n_stocks, n_bad):
    ratio = math.exp(2)
    return {
        "min_single_stock_weight": 1 / (1 + n_stocks * ratio),
        "max_cash_weight": ratio / (n_stocks + ratio),
        "min_total_bad_stock_weight": n_bad / (n_bad + (n_stocks - n_bad + 1) * ratio),
    }


def bad_stock_share(stock_weights, bad_labels):
    weights = np.asarray(stock_weights, dtype=float)
    labels = np.asarray(bad_labels, dtype=bool)
    if weights.shape != labels.shape or weights.ndim != 2:
        raise ValueError("Stock weights and labels must be aligned weekly matrices")
    invested = weights.sum(axis=1)
    if np.any(invested <= 0):
        raise ValueError("Bad-stock share requires positive stock exposure")
    actual = (weights * labels).sum(axis=1) / invested
    equal_weight = labels.mean(axis=1)
    return actual, equal_weight


def conditional_bootstrap(delta_by_week, condition, *, draws=2000, block=8, seed=42):
    values = np.asarray(delta_by_week, dtype=float)
    mask = np.asarray(condition, dtype=bool)
    if values.shape != mask.shape or values.ndim != 1:
        raise ValueError("Paired deltas and condition must align by week")
    if not np.all(np.isfinite(values)):
        raise ValueError("Paired deltas must be finite")
    n = len(values)
    if not mask.any():
        return {
            "weeks": 0,
            "mean": None,
            "lower_95": None,
            "upper_95": None,
            "valid_draws": 0,
        }
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(draws, math.ceil(n / block)))
    indices = ((starts[:, :, None] + np.arange(block)) % n).reshape(draws, -1)[:, :n]
    selected = mask[indices]
    counts = selected.sum(axis=1)
    valid = counts > 0
    sampled_means = (values[indices] * selected).sum(axis=1)[valid] / counts[valid]
    low, high = np.quantile(sampled_means, [0.025, 0.975])
    return {
        "weeks": int(mask.sum()),
        "mean": float(values[mask].mean()),
        "lower_95": float(low),
        "upper_95": float(high),
        "valid_draws": int(valid.sum()),
    }

"""Research statistics; practical equality is distinct from exact equality."""

import numpy as np
from scipy.stats import spearmanr


def dispersion(weights):
    values = np.asarray(weights, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid stock weights")
    if len(values) < 2 or values.sum() <= 0:
        return None  # Cash-only and one-stock portfolios cannot test dispersion.
    shares = values / values.sum()
    relative = len(shares) * shares - 1
    return {
        "cv": float(np.std(relative)),
        "max_relative_deviation": float(np.abs(relative).max()),
        "equity_tv": float(np.abs(shares - 1 / len(shares)).sum() / 2),
        "effective_names_fraction": float(1 / (len(shares) * np.sum(shares**2))),
        "max_min_ratio": float(shares.max() / shares.min()),
    }


def rank_correlation(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 3 or np.ptp(a) <= 1e-12 or np.ptp(b) <= 1e-12:
        return None
    return float(spearmanr(a, b).statistic)


def bootstrap(values, block=4, seed=42, n=10000):
    """Moving blocks preserve local serial dependence; intervals are exploratory."""
    x = np.asarray(values, dtype=float)
    if len(x) < block or not np.isfinite(x).all():
        raise ValueError("Insufficient finite observations")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(x) - block + 1, (n, int(np.ceil(len(x) / block))))
    samples = x[starts[..., None] + np.arange(block)].reshape(n, -1)[:, : len(x)]
    means = samples.mean(axis=1)
    return {
        "mean": float(x.mean()),
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        "upper95_one_sided": float(np.quantile(means, 0.95)),
        "block_weeks": block,
    }


def dispersion_summary(rows):
    valid = [r for r in rows if r["k"] >= 2]
    if not valid:
        raise ValueError("No informative multi-stock weeks")
    result = {
        "weeks": len(rows),
        "informative_weeks": len(valid),
        "cash_only_weeks": sum(r["k"] == 0 for r in rows),
        "one_stock_weeks": sum(r["k"] == 1 for r in rows),
    }
    for key in [
        "cv",
        "max_relative_deviation",
        "equity_tv",
        "effective_names_fraction",
    ]:
        x = np.array([r[key] for r in valid])
        result[key] = {
            "mean": float(x.mean()),
            "median": float(np.median(x)),
            "min": float(x.min()),
            "max": float(x.max()),
            "p95": float(np.quantile(x, 0.95)),
        }
    result["threshold_sensitivity"] = {
        str(t): {
            "fraction_weeks_all_names_within": float(
                np.mean([r["max_relative_deviation"] <= t for r in valid])
            ),
            "fraction_weeks_cv_within": float(np.mean([r["cv"] <= t for r in valid])),
        }
        for t in [0.01, 0.025, 0.05, 0.10, 0.15, 0.20]
    }
    result["mean_cv_bootstrap"] = {
        str(b): bootstrap([r["cv"] for r in valid], block=b)
        for b in [4, 8, 13]
        if len(valid) >= b
    }
    return result

"""Weekly inference and chronology helpers for feature-return research."""

import numpy as np
import pandas as pd


def forward_daily(close, invalid_steps):
    """Label each close with the next exchange session, preserving audit gaps."""
    returns = close.div(close.shift()).sub(1)
    for day, symbol in invalid_steps:
        if day in returns.index and symbol in returns.columns:
            returns.loc[day, symbol] = np.nan
    return returns.shift(-1), close.index.to_series().shift(-1)


def percentile_groups(values):
    """Map average ranks to five buckets without splitting identical values."""
    percentile = (values.rank(method="average") - 0.5) / values.count()
    return np.floor(percentile * 5).clip(0, 4).add(1)


def calendar_matrix(values):
    """Preserve unavailable Fridays instead of joining observations across gaps."""
    frame = values.to_frame() if isinstance(values, pd.Series) else values
    if frame.index.has_duplicates or (frame.index.dayofweek != 4).any():
        raise ValueError("Expected unique calendar Fridays")
    grid = pd.date_range(frame.index.min(), frame.index.max(), freq="W-FRI")
    return frame.reindex(grid).to_numpy(dtype=float)


def bootstrap_mean(matrix, *, block=4, reps=10_000, seed=42):
    """Percentile intervals and centered two-sided p-values over calendar weeks."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    observed = np.isfinite(values)
    counts = observed.sum(axis=0)
    total = np.where(observed, values, 0)
    means = np.divide(
        total.sum(axis=0),
        counts,
        where=counts > 0,
        out=np.full(values.shape[1], np.nan),
    )
    n = len(values)
    if n < block * 2:
        raise ValueError("Too few calendar weeks for block inference")
    rng = np.random.default_rng(seed)
    draws = np.empty((reps, values.shape[1]))
    # Aggregate sampled week multiplicities to keep memory independent of columns.
    for first in range(0, reps, 200):
        size = min(200, reps - first)
        starts = rng.integers(0, n, size=(size, (n + block - 1) // block))
        index = ((starts[..., None] + np.arange(block)) % n).reshape(size, -1)[:, :n]
        weights = np.zeros((size, n))
        np.add.at(weights, (np.arange(size)[:, None], index), 1)
        divisor = weights @ observed.astype(float)
        draws[first : first + size] = np.divide(
            weights @ total,
            divisor,
            out=np.full_like(divisor, np.nan),
            where=divisor > 0,
        )
    rows = []
    for i, mean in enumerate(means):
        valid = draws[:, i][np.isfinite(draws[:, i])]
        if counts[i] < 12 or len(valid) < reps * 0.9:
            rows.append((mean, np.nan, np.nan, np.nan, int(counts[i])))
            continue
        low, high = np.quantile(valid, [0.025, 0.975])
        p = (1 + np.count_nonzero(np.abs(valid - mean) >= abs(mean))) / (len(valid) + 1)
        rows.append((mean, low, high, p, int(counts[i])))
    return pd.DataFrame(rows, columns=["mean", "ci_low", "ci_high", "p_value", "weeks"])


def holm(pvalues):
    """Adjust a family of p-values while preserving names and missing values."""
    ordered = pvalues.dropna().sort_values()
    adjusted = (ordered * np.arange(len(ordered), 0, -1)).cummax().clip(upper=1)
    return adjusted.reindex(pvalues.index)


def training_rows(panel, refit, horizon):
    """Select the trailing five-year fit using only fully matured outcomes."""
    dates = panel.index.get_level_values("friday")
    return panel.loc[
        (dates >= refit - pd.DateOffset(years=5))
        & panel[f"{horizon}_end"].lt(refit)
        & panel[horizon].notna()
    ]


def select_basket(rows, score, horizon, *, k=15):
    """Select by forecast and ticker, then inspect labels without replacement."""
    ordered = rows.sort_index().sort_values(score, ascending=False, kind="stable")
    picks = ordered.index[:k].tolist()
    if len(picks) != k:
        raise ValueError("Insufficient scoring candidates")
    actual = ordered.iloc[:k][horizon].mean(skipna=False)
    return picks, actual


def weekly_relationships(panel, features):
    """Create paired stock-rank and feature-group observations per Friday."""
    rows, groups = [], []
    for day, week in panel.groupby(level="friday", sort=True):
        for feature in features:
            bucket = percentile_groups(week[feature])
            for horizon in ("daily", "weekly"):
                paired = week[[feature, horizon]].dropna()
                if len(paired) < 30 or paired[horizon].nunique() < 2:
                    continue
                ic = (
                    paired[feature].corr(paired[horizon], method="spearman")
                    if paired[feature].nunique() > 1
                    else np.nan
                )
                labeled = pd.DataFrame({"group": bucket, "actual": week[horizon]})
                means = labeled.groupby("group")["actual"].mean().reindex(range(1, 6))
                sizes = (
                    labeled.dropna()
                    .groupby("group")
                    .size()
                    .reindex(range(1, 6), fill_value=0)
                )
                for group in range(1, 6):
                    groups.append(
                        {
                            "friday": day,
                            "feature": feature,
                            "horizon": horizon,
                            "group": group,
                            "return": means[group],
                            "stocks": int(sizes[group]),
                        }
                    )
                rows.append(
                    {
                        "friday": day,
                        "feature": feature,
                        "horizon": horizon,
                        "ic": ic,
                        "spread": means[5] - means[1],
                        "curvature": means[1] - 2 * means[3] + means[5],
                        "stocks": len(paired),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(groups)


def portfolio_observations(predictions, arms, horizon):
    """Aggregate errors and preselected baskets without future eligibility checks."""
    rows, picks = [], []
    for day, week in predictions.groupby(level="friday", sort=True):
        week = week.droplevel("friday")
        observed = week[horizon].notna()
        for arm in arms:
            chosen, actual = select_basket(week, arm, horizon)
            paired = week.loc[observed, [arm, horizon]]
            ic = (
                paired[arm].corr(paired[horizon], method="spearman")
                if paired[arm].nunique() > 1
                else np.nan
            )
            rows.append(
                {
                    "friday": day,
                    "horizon": horizon,
                    "arm": arm,
                    "mae": (paired[arm] - paired[horizon]).abs().mean(),
                    "ic": ic,
                    "basket_return": actual,
                    "stocks": int(observed.sum()),
                    "scored": len(week),
                }
            )
            for symbol in chosen:
                picks.append(
                    {
                        "friday": day,
                        "horizon": horizon,
                        "arm": arm,
                        "symbol": symbol,
                        "actual": week.loc[symbol, horizon],
                        "forecast": week.loc[symbol, arm],
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(picks)


def close_cost_diagnostic(picks, close, *, bps=10):
    """Drift each basket to the next decision and budget proportional trade costs."""
    rows = []
    for arm, model in picks.loc[picks.horizon.eq("weekly")].groupby("arm"):
        previous = None
        last_prices = None
        for day, week in model.groupby("friday", sort=True):
            target = pd.Series(1 / 15, index=week.symbol)
            symbols = (
                target.index if previous is None else target.index.union(previous.index)
            )
            prices = close.loc[day, symbols]
            if prices.isna().any() or prices.le(0).any():
                rows.append(
                    {
                        "friday": day,
                        "arm": arm,
                        "turnover": np.nan,
                        "net_return": np.nan,
                    }
                )
                previous, last_prices = None, None
                continue
            current = pd.Series(0.0, index=symbols)
            if previous is not None:
                drift = previous * prices[previous.index] / last_prices
                current.loc[previous.index] = drift / drift.sum()
            weights = target.reindex(symbols, fill_value=0)
            low, high = 0.0, 1.0
            for _ in range(40):
                invested = (low + high) / 2
                turnover = (invested * weights - current).abs().sum()
                if invested + bps / 10_000 * turnover <= 1:
                    low = invested
                else:
                    high = invested
            turnover = (low * weights - current).abs().sum()
            actual = week.actual.mean(skipna=False)
            rows.append(
                {
                    "friday": day,
                    "arm": arm,
                    "turnover": turnover,
                    "net_return": low * (1 + actual) - 1,
                }
            )
            previous, last_prices = target, prices[target.index]
    return pd.DataFrame(rows)

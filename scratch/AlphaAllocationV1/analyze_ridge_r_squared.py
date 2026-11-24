"""Measure archived ridge R-squared and paired weekly squared-error differences."""

import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

DATA = Path(__file__).resolve().parent / "data/ridge_lookback"
ARMS = ("One channel", "Seven channels", "Zero")
REPLICATES = 20_000


def paired_difference(values, block):
    """Circular blocks preserve pairing and adjacent observed-week dependence."""
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or len(values) <= block:
        raise ValueError("Need finite paired differences and more weeks than a block")
    rng = np.random.default_rng(42)
    starts = rng.integers(
        0, len(values), size=(REPLICATES, (len(values) + block - 1) // block)
    )
    indices = ((starts[..., None] + np.arange(block)) % len(values)).reshape(
        REPLICATES, -1
    )[:, : len(values)]
    draws = values[indices].mean(axis=1)
    mean = values.mean()
    low, high = np.quantile(draws, [0.025, 0.975])
    p = (1 + np.count_nonzero(np.abs(draws - mean) >= abs(mean))) / (REPLICATES + 1)
    return {
        "mean_seven_minus_one_pp_squared": float(mean),
        "ci95_pp_squared": [float(low), float(high)],
        "two_sided_p": float(p),
        "block_observed_weeks": block,
    }


def spy_comparison(weekly):
    """Describe market co-movement separately from direct SPY tracking error."""
    if weekly.duplicated(["friday", "arm"]).any():
        raise ValueError("Duplicate weekly forecasts")
    one = weekly.loc[weekly.arm.eq("One channel")].set_index("friday").sort_index()
    seven = weekly.loc[weekly.arm.eq("Seven channels")].set_index("friday").sort_index()
    pd.testing.assert_frame_equal(
        one[["spy", "basket_actual"]], seven[["spy", "basket_actual"]]
    )
    spy = one.spy.to_numpy()
    design = np.column_stack([np.ones(len(spy)), spy])
    if len(spy) <= 2 or np.linalg.matrix_rank(design) != 2:
        raise ValueError("Market regression requires varying SPY returns and n > 2")
    records = []
    for name, returns in (
        ("One-channel basket forecast", one.basket_forecast.to_numpy()),
        ("Seven-channel basket forecast", seven.basket_forecast.to_numpy()),
        ("Actual fixed basket", one.basket_actual.to_numpy()),
    ):
        if not np.isfinite(returns).all() or np.ptp(returns) == 0:
            raise ValueError("Market regression requires finite, varying returns")
        intercept, beta = np.linalg.lstsq(design, returns, rcond=None)[0]
        fitted = intercept + beta * spy
        r_squared = float(r2_score(returns, fitted))
        records.append(
            {
                "series": name,
                "n": len(spy),
                "market_regression_r_squared": r_squared,
                "market_regression_adjusted_r_squared": 1
                - (1 - r_squared) * (len(spy) - 1) / (len(spy) - 2),
                "intercept_pp_per_week": float(100 * intercept),
                "spy_beta": float(beta),
                "direct_spy_tracking_r_squared": float(r2_score(spy, returns)),
            }
        )
    return records


def main():
    stock_path = DATA / "stock_forecasts.parquet"
    weekly_path = DATA / "weekly_forecasts.csv"
    with duckdb.connect() as db:
        stocks = db.read_parquet(str(stock_path)).df()
    weekly = pd.read_csv(weekly_path, parse_dates=["friday"])
    if stocks.duplicated(["friday", "symbol"]).any():
        raise ValueError("Duplicate stock-week outcomes")
    if stocks[["actual", *ARMS]].isna().any().any():
        raise ValueError("Missing actuals or forecasts")
    records = []
    for arm in ARMS:
        basket = weekly.loc[weekly.arm.eq(arm)].set_index("friday").sort_index()
        for level, actual, predicted in (
            ("Individual stock", stocks.actual, stocks[arm]),
            ("15-stock basket", basket.basket_actual, basket.basket_forecast),
        ):
            records.append(
                {
                    "level": level,
                    "arm": arm,
                    "n": len(actual),
                    "r_squared": float(r2_score(actual, predicted)),
                    "rmse_pp": float(100 * np.sqrt(np.mean((actual - predicted) ** 2))),
                    "adjusted_r_squared": None,
                }
            )
    metrics = pd.DataFrame(records)
    metrics.to_csv(DATA / "r_squared.csv", index=False)
    market = spy_comparison(weekly)
    pd.DataFrame(market).to_csv(DATA / "spy_r_squared.csv", index=False)

    # Average stock losses within a week before comparing the two arms.
    stock_mse = (
        pd.DataFrame({arm: (100 * (stocks[arm] - stocks.actual)) ** 2 for arm in ARMS})
        .groupby(stocks.friday)
        .mean()
    )
    basket_mse = weekly.assign(
        mse_pp_squared=(100 * (weekly.basket_forecast - weekly.basket_actual)) ** 2
    ).pivot(index="friday", columns="arm", values="mse_pp_squared")
    deltas = pd.DataFrame(
        {
            "stock_mse_difference_pp_squared": stock_mse["Seven channels"]
            - stock_mse["One channel"],
            "basket_mse_difference_pp_squared": basket_mse["Seven channels"]
            - basket_mse["One channel"],
        }
    ).sort_index()
    if deltas.isna().any().any() or len(deltas) != 33:
        raise ValueError("Expected the same 33 matched weeks at both levels")
    deltas.to_csv(DATA / "squared_error_weekly.csv")
    comparisons = {
        level: {
            str(block): paired_difference(deltas[column], block) for block in (2, 4, 8)
        }
        for level, column in (
            ("stock", "stock_mse_difference_pp_squared"),
            ("basket", "basket_mse_difference_pp_squared"),
        )
    }
    primary = pd.Series(
        {key: value["4"]["two_sided_p"] for key, value in comparisons.items()}
    ).sort_values()
    adjusted = (primary * np.arange(len(primary), 0, -1)).cummax().clip(upper=1)
    for level, p in adjusted.items():
        comparisons[level]["4"]["holm_p_two_levels"] = float(p)

    with np.load(DATA / "ridge_weights.npz") as weights:
        coefficients = {
            "One channel": int(weights["one_weights"].size),
            "Seven channels": int(weights["seven_weights"].size),
        }
    summary = {
        "metrics": records,
        "spy_comparison": market,
        "spy_comparison_method": (
            "Fit series_return = intercept + beta * realized_SPY_return by OLS "
            "on the same 33 weeks. Adjusted R-squared uses one fitted slope. "
            "This is descriptive in-sample market co-movement for archived "
            "forecasts and the single shared actual basket. Direct SPY tracking "
            "R-squared instead scores the unmodified series with SPY as target."
        ),
        "lag_coefficients_excluding_intercept": coefficients,
        "comparison": comparisons,
        "adjusted_r_squared_reason": (
            "Classical adjusted R-squared is an in-sample OLS statistic. These "
            "are test forecasts from regularized ridge models. The inputs "
            "flatten to 255 or 1785 lag coefficients. Ridge effective degrees "
            "of freedom require the training design and penalty. The basket "
            "contains 33 outcomes and is aggregated from stock predictions."
        ),
        "anova_reason": (
            "Nested OLS ANOVA assumptions do not hold for these separately "
            "regularized, non-nested feature recipes and dependent stock-weeks. "
            "Compare paired squared forecast errors in weekly blocks instead."
        ),
        "bootstrap": {
            "replicates": REPLICATES,
            "seed": 42,
            "primary_block": 4,
            "sensitivity_blocks": [2, 8],
        },
        "scope": "Exploratory reuse of the archived 33 historical evaluation weeks",
        "sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                stock_path,
                weekly_path,
                DATA / "ridge_weights.npz",
                Path(__file__),
            )
        },
    }
    (DATA / "r_squared_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(metrics.to_string(index=False))
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()

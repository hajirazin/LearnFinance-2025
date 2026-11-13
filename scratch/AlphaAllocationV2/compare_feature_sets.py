"""Compare equal-weight top-15 ridge screens on paired 2025 Fridays."""

import argparse
import hashlib
import json
import subprocess
from itertools import combinations
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from build_friday_dataset import (
    FEATURES,
    OUTPUT,
    join_friday_ohlcv,
    plt,
    weekly_spearman,
)
from matplotlib.ticker import PercentFormatter
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FIVE = (
    "momentum_20d",
    "momentum_12_1",
    "volatility_20d",
    "relative_volume_20d",
    "news_sentiment",
)
SEVEN = (*FIVE, "close", "volume")
SIX = (*FIVE, "close")
EVAL_START = pd.Timestamp("2025-01-01")
EVAL_END = pd.Timestamp("2025-12-31")
TRAIN_YEARS = 5
RIDGE_ALPHA = 1.0
TOP_K = 15
BOOTSTRAP_SEED = 42
BLOCK_WEEKS = 4
BOOTSTRAP_REPS = 5000


def score_fridays(panel, arms):
    calendar = pd.DatetimeIndex(panel.index.get_level_values("friday").unique())
    decision = panel.index.get_level_values("friday")
    required_features = sorted(
        {feature for features in arms.values() for feature in features}
    )
    common = panel.dropna(subset=required_features)
    common = common.loc[common["price_usable_at_cutoff"]]
    complete_labels = common.loc[common["forward_return"].notna()]
    train_dates = complete_labels.index.get_level_values("friday")
    dates = calendar[calendar.to_series().between(EVAL_START, EVAL_END).to_numpy()]
    dates = dates[dates + pd.Timedelta(days=7) <= EVAL_END]
    dates = dates[(dates + pd.Timedelta(days=7)).isin(calendar)]
    scored = []
    fits = []
    for month in dates.to_period("M").unique():
        month_dates = dates[dates.to_period("M") == month]
        refit = month_dates.min()
        train_start = refit - pd.DateOffset(years=TRAIN_YEARS)
        train = complete_labels.loc[
            (train_dates >= train_start) & complete_labels["next_friday"].lt(refit)
        ]
        week_rows = common.loc[
            common.index.get_level_values("friday").isin(month_dates)
        ]
        if (
            not week_rows.groupby(level="friday")
            .size()
            .reindex(month_dates, fill_value=0)
            .ge(TOP_K)
            .all()
        ):
            raise ValueError(f"Insufficient candidates in {month}")
        scores = pd.DataFrame(index=week_rows.index)
        for name, features in arms.items():
            model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
            model.fit(train[list(features)], train["forward_return"])
            scores[name] = model.predict(week_rows[list(features)])
        scored.append(scores)
        fits.append(
            {
                "refit_friday": refit.date().isoformat(),
                "train_start": train_start.date().isoformat(),
                "last_training_label": train["next_friday"].max().date().isoformat(),
                "training_rows": len(train),
            }
        )
    predictions = pd.concat(scored).sort_index()
    labels = panel.loc[decision >= EVAL_START, ["next_friday", "forward_return"]]
    return predictions.join(labels), fits


def evaluate(predictions, arms):
    metrics = predictions.groupby(level="friday")["forward_return"].agg(
        universe_return="mean",
        scorable_stocks="size",
        stocks_with_return="count",
    )
    picks = []
    stock_forecasts = []
    for name in arms:
        # Alphabetical ticker breaks score ties and makes the selection reproducible.
        ranked = predictions.reset_index().sort_values(
            ["friday", name, "symbol"], ascending=[True, False, True]
        )
        ranked["rank"] = ranked.groupby("friday", sort=False).cumcount() + 1
        ranked = ranked.rename(columns={name: "predicted_return"}).assign(model=name)
        ranked["prediction_error"] = (
            ranked["predicted_return"] - ranked["forward_return"]
        )
        columns = [
            "friday",
            "model",
            "rank",
            "symbol",
            "predicted_return",
            "forward_return",
        ]
        stock_forecasts.append(ranked[columns + ["next_friday", "prediction_error"]])
        basket = ranked.loc[ranked["rank"].le(TOP_K)]
        picks.append(basket[columns])
        metrics[f"{name}_top15_predicted_return"] = basket.groupby("friday")[
            "predicted_return"
        ].mean()
        realized = basket.groupby("friday")["forward_return"]
        metrics[f"{name}_missing_picks"] = TOP_K - realized.count()
        metrics[f"{name}_top15_return"] = realized.mean().where(
            realized.count().eq(TOP_K)
        )
        metrics[f"{name}_top15_prediction_error"] = (
            metrics[f"{name}_top15_predicted_return"] - metrics[f"{name}_top15_return"]
        )
        metrics[f"{name}_rank_ic"] = weekly_spearman(
            predictions, name, "forward_return"
        )["spearman"]
    metrics["paired"] = (
        metrics[[f"{name}_top15_return" for name in arms]].notna().all(axis=1)
    )
    stock_forecasts = pd.concat(stock_forecasts, ignore_index=True)
    stock_forecasts["paired"] = stock_forecasts["friday"].map(metrics["paired"])
    return metrics, pd.concat(picks, ignore_index=True), stock_forecasts


def forecast_accuracy(predicted, actual):
    observed = predicted.notna() & actual.notna()
    predicted, actual = predicted[observed], actual[observed]
    error = predicted - actual
    return {
        "observations": len(error),
        "mean_predicted_return": predicted.mean(),
        "mean_actual_return": actual.mean(),
        "mean_error": error.mean(),
        "mae": error.abs().mean(),
        "rmse": np.sqrt(error.pow(2).mean()),
        "zero_forecast_mae": actual.abs().mean(),
    }


def plot_basket_forecasts(metrics, arms, output, *, titles=None):
    fig, axes = plt.subplots(
        len(arms), 1, sharex=True, sharey=True, figsize=(12, 2.7 * len(arms))
    )
    for name, ax in zip(arms, axes, strict=True):
        paired = metrics["paired"]
        actual = metrics[f"{name}_top15_return"].where(paired)
        predicted = metrics[f"{name}_top15_predicted_return"].where(paired)
        ax.axhline(0, color="0.75", linewidth=0.8)
        ax.plot(actual, "o-", color="#175e84", markersize=3, label="Actual")
        ax.plot(predicted, "o-", color="#bc581c", markersize=3, label="Predicted")
        title = f"{len(arms[name])} features" if titles is None else titles[name]
        ax.set_title(title, loc="left")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    axes[0].legend(loc="upper right", ncols=2, frameon=False)
    axes[-1].set_xlabel("Decision Friday in 2025")
    fig.suptitle(f"Predicted and actual equal-weight top-{TOP_K} returns", fontsize=16)
    fig.supylabel("Friday-to-Friday return")
    fig.tight_layout()
    fig.savefig(output / "basket_predicted_vs_actual.png", dpi=160)
    plt.close(fig)


def interval(differences):
    values = differences.to_numpy()
    if len(values) < BLOCK_WEEKS or not np.isfinite(values).all():
        raise ValueError("Insufficient finite paired returns for block bootstrap")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    block_count = (len(values) + BLOCK_WEEKS - 1) // BLOCK_WEEKS
    starts = rng.integers(
        0, len(values) - BLOCK_WEEKS + 1, size=(BOOTSTRAP_REPS, block_count)
    )
    indices = (starts[..., None] + np.arange(BLOCK_WEEKS)).reshape(BOOTSTRAP_REPS, -1)
    means = values[indices[:, : len(values)]].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--seven-vs-eight", action="store_true")
    mode.add_argument("--six-seven-eight", action="store_true")
    args = parser.parse_args()
    if args.six_seven_eight:
        arms = {"six": SIX, "seven": SEVEN, "eight": tuple(FEATURES)}
    elif args.seven_vs_eight:
        arms = {"seven": SEVEN, "eight": tuple(FEATURES)}
    else:
        arms = {"five": FIVE, "eight": tuple(FEATURES)}
    uses_ohlcv = args.six_seven_eight or args.seven_vs_eight
    source = OUTPUT / "friday_dataset.parquet"
    with duckdb.connect() as db:
        panel = (
            db.read_parquet(str(source))
            .df()
            .set_index(["friday", "symbol"])
            .sort_index()
        )
    if not panel.index.is_unique:
        raise ValueError("Duplicate Friday/symbol keys")
    if uses_ohlcv:
        panel = join_friday_ohlcv(panel)
    predictions, fits = score_fridays(panel, arms)
    metrics, picks, stock_forecasts = evaluate(predictions, arms)
    paired = metrics.loc[metrics["paired"]]
    with source.open("rb") as handle:
        source_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    summary = {
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "friday_dataset_sha256": source_hash,
        "parameters": {
            "training_years": TRAIN_YEARS,
            "ridge_alpha": RIDGE_ALPHA,
            "top_k": TOP_K,
            "bootstrap_block_weeks": BLOCK_WEEKS,
            "bootstrap_reps": BOOTSTRAP_REPS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "features": arms,
        "scored_weeks": len(metrics),
        "paired_weeks": len(paired),
        "omitted_weeks": metrics.index[~metrics["paired"]]
        .strftime("%Y-%m-%d")
        .tolist(),
        "paired_date_range": [
            paired.index.min().date().isoformat(),
            paired.index.max().date().isoformat(),
        ],
        "median_scorable_stocks": paired["scorable_stocks"].median(),
        "mean_universe_return": paired["universe_return"].mean(),
        "refits": fits,
        "forecast_accuracy": {
            "basis": "Paired Fridays. Stock metrics use all scored stocks with observed returns.",
            "error_definition": "predicted_return minus actual_return",
            "stock_forecast_rows": len(stock_forecasts),
            "by_model": {},
        },
    }
    for left, right in combinations(arms, 2):
        difference = paired[f"{left}_top15_return"] - paired[f"{right}_top15_return"]
        summary.update(
            {
                f"mean_{left}_minus_{right}": difference.mean(),
                f"{left}_minus_{right}_ci95": interval(difference),
                f"weeks_{left}_beat_{right}": int(difference.gt(0).sum()),
                f"weeks_{right}_beat_{left}": int(difference.lt(0).sum()),
            }
        )
    for name in arms:
        returns = paired[f"{name}_top15_return"]
        stocks = stock_forecasts.loc[
            stock_forecasts["model"].eq(name) & stock_forecasts["paired"]
        ]
        summary["forecast_accuracy"]["by_model"][name] = {
            "stock": forecast_accuracy(
                stocks["predicted_return"], stocks["forward_return"]
            ),
            "basket": forecast_accuracy(
                paired[f"{name}_top15_predicted_return"], returns
            ),
        }
        sets = (
            picks.loc[picks["model"].eq(name) & picks["friday"].isin(paired.index)]
            .groupby("friday")["symbol"]
            .agg(set)
        )
        summary[name] = {
            "mean_top15_return": returns.mean(),
            "mean_top15_excess": (returns - paired["universe_return"]).mean(),
            "mean_rank_ic": paired[f"{name}_rank_ic"].mean(),
            "mean_fraction_replaced": float(
                np.mean(
                    [
                        len(current - previous) / TOP_K
                        for previous, current in zip(
                            sets.iloc[:-1], sets.iloc[1:], strict=True
                        )
                    ]
                )
            ),
        }
    output = OUTPUT / "feature_comparison"
    if uses_ohlcv:
        output /= "six_seven_eight" if args.six_seven_eight else "seven_vs_eight"
        with (OUTPUT.parent / "daily_ohlcv_2016_2025.csv").open("rb") as handle:
            summary["daily_ohlcv_sha256"] = hashlib.file_digest(
                handle, "sha256"
            ).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "weekly_results.csv")
    picks.to_csv(output / "top_15_picks.csv", index=False)
    with duckdb.connect() as db:
        db.from_df(stock_forecasts).write_parquet(
            str(output / "stock_forecasts.parquet")
        )
    plot_basket_forecasts(metrics, arms, output)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

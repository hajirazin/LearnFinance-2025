"""Compare one-year, one- and seven-channel ridge forecasts on matched weeks."""

import json
import subprocess
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from build_dataset import ROSTER
from compare_patchtst import sha256
from compare_v1_forecast_accuracy import decorate, line, paired_interval, plt, set_style
from run_v1_h28 import FORECAST_EVIDENCE
from scipy.linalg import eigh
from v1_hypothesis_data import (
    DATA,
    SEVEN,
    VALIDATION,
    WIDTH,
    daily_features,
    decision_dates,
    session_data,
    windows,
)

ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "data/ridge_lookback"
FIGURE = ROOT / "figures/ridge/one_year_feature_comparison.png"
PENALTIES = np.array([0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0])
LEARNED = ("One channel", "Seven channels")


def design(x, mask):
    """Flatten the same observed stock histories for the two regressions."""
    observed = mask.sum(axis=1, keepdims=True)
    if not observed.min():
        raise ValueError("Every feature needs observed inputs")
    mean = (x * mask).sum(axis=1, keepdims=True, dtype=np.float64) / observed
    centered = np.where(mask, x - mean, 0.0)
    scale = np.sqrt((centered**2).sum(axis=1, keepdims=True) / observed + 1e-5)
    normalized = centered / scale
    return normalized[:, :, -1].copy(), normalized[:, :, : len(SEVEN)].reshape(
        len(x), -1
    ).copy()


def batches(logs, features, dates, *, labeled):
    for start in range(0, len(dates), 10):
        index, x, mask, y = windows(
            logs, features, dates[start : start + 10], use_features=True, year=True
        )
        if labeled:
            valid = np.isfinite(y).all(axis=1)
            index, x, mask, y = index[valid], x[valid], mask[valid], y[valid]
            target = np.expm1(y.sum(axis=1, dtype=np.float64))
        else:
            target = None
        yield index, design(x, mask), target


class RidgeMoments:
    """Pool sufficient statistics so ridge does not keep all windows in memory."""

    def __init__(self, columns):
        self.n = 0
        self.x = np.zeros(columns)
        self.y = 0.0
        self.xx = np.zeros((columns, columns))
        self.xy = np.zeros(columns)

    def update(self, x, y):
        self.n += len(y)
        self.x += x.sum(axis=0)
        self.y += y.sum()
        self.xx += x.T @ x
        self.xy += x.T @ y

    def coefficients(self, penalties):
        x_mean = self.x / self.n
        y_mean = self.y / self.n
        covariance = self.xx / self.n - np.outer(x_mean, x_mean)
        xy = self.xy / self.n - x_mean * y_mean
        # Solve the average squared-error ridge objective for all penalties.
        eigenvalues, vectors = eigh(covariance, check_finite=True)
        coefficients = vectors @ (
            (vectors.T @ xy)[:, None]
            / (np.maximum(eigenvalues, 0.0)[:, None] + penalties[None, :])
        )
        return coefficients, y_mean - x_mean @ coefficients


def fitted_models(logs, features, train_dates, validation_dates):
    moments = [RidgeMoments(WIDTH), RidgeMoments(WIDTH * len(SEVEN))]
    symbols = set()
    for index, inputs, target in batches(logs, features, train_dates, labeled=True):
        symbols.update(index.get_level_values("symbol"))
        for stats, matrix in zip(moments, inputs, strict=True):
            stats.update(matrix, target)
    rows = moments[0].n
    print(
        f"Fit both arms on {rows:,} labeled examples from {len(symbols)} stocks.",
        flush=True,
    )
    candidates = [stats.coefficients(PENALTIES) for stats in moments]
    validation = [[], []]
    n_validation = 0
    for index, inputs, target in batches(
        logs, features, validation_dates, labeled=True
    ):
        n_validation += len(index)
        for errors, matrix, (weights, bias) in zip(
            validation, inputs, candidates, strict=True
        ):
            errors.append(
                pd.DataFrame(
                    np.abs(matrix @ weights + bias - target[:, None]), index=index
                )
            )
    models, settings = {}, {}
    for name, errors, (weights, bias) in zip(
        LEARNED, validation, candidates, strict=True
    ):
        mae = pd.concat(errors).groupby(level="friday").mean().mean() * 100
        best = int(mae.idxmin())
        models[name] = weights[:, best], bias[best]
        settings[name] = {
            "penalty_on_average_mse": float(PENALTIES[best]),
            "validation_stock_mae_pp": float(mae.iloc[best]),
            "validation_grid_mae_pp": {
                str(value): float(score)
                for value, score in zip(PENALTIES, mae, strict=True)
            },
        }
    return (
        models,
        settings,
        {
            "training_rows": rows,
            "training_symbols": len(symbols),
            "validation_rows": n_validation,
        },
    )


def score(logs, features, dates, models):
    predictions = []
    for index, matrices, _ in batches(logs, features, dates, labeled=False):
        predictions.append(
            pd.DataFrame(
                {
                    name: matrix @ models[name][0] + models[name][1]
                    for name, matrix in zip(LEARNED, matrices, strict=True)
                },
                index=index,
            )
        )
    scores = pd.concat(predictions).sort_index()
    if not np.isfinite(scores).all().all():
        raise ValueError("Nonfinite ridge forecasts")
    return scores


def evaluation(scores):
    with duckdb.connect() as db:
        panel = db.read_parquet(str(FORECAST_EVIDENCE / "stock_forecasts.parquet")).df()
    panel["friday"] = panel.friday.dt.as_unit("ns")
    panel = panel.set_index(["friday", "symbol"])[["actual"]].dropna()
    panel = panel.join(scores, how="left", validate="one_to_one").assign(Zero=0.0)
    if panel.isna().any().any():
        raise ValueError("Ridge scores do not cover the archived common outcomes")
    fixed = pd.read_csv(
        FORECAST_EVIDENCE / "weekly_forecasts.csv", parse_dates=["friday"]
    )
    fixed = fixed.loc[fixed.method.eq("Native") & fixed.model.eq("V1")].set_index(
        "friday"
    )
    fixed.index = fixed.index.as_unit("ns")
    if not fixed.index.equals(panel.index.get_level_values("friday").unique()):
        raise ValueError("Evaluation weeks drifted from the archive")
    weekly = []
    for day, group in panel.groupby(level="friday", sort=True):
        names = fixed.loc[day, "fixed_symbols"].split(",")
        basket = group.droplevel("friday").loc[names]
        if len(basket) != 15 or not np.isclose(
            basket.actual.mean(), fixed.loc[day, "fixed_actual"], atol=1e-12, rtol=0
        ):
            raise ValueError(f"Basket labels drifted from the archive on {day.date()}")
        for arm in (*LEARNED, "Zero"):
            error = 100 * (group[arm] - group.actual)
            weekly.append(
                {
                    "friday": day,
                    "arm": arm,
                    "matched_stocks": len(group),
                    "stock_mae_pp": error.abs().mean(),
                    "rank_ic": group[arm].rank().corr(group.actual.rank())
                    if arm in LEARNED
                    else np.nan,
                    "basket_forecast": basket[arm].mean(),
                    "basket_actual": basket.actual.mean(),
                    "basket_mae_pp": 100 * abs((basket[arm] - basket.actual).mean()),
                    "spy": fixed.loc[day, "spy"],
                }
            )
    return panel, pd.DataFrame(weekly)


def compare(weekly, field):
    paired = weekly.pivot(index="friday", columns="arm", values=field)
    delta = paired["Seven channels"] - paired["One channel"]
    mean, low, high, p = paired_interval(delta)
    return {
        "seven_minus_one_pp": mean,
        "ci95_pp": [low, high],
        "one_sided_p_for_lower_seven_error": p,
        "weeks_seven_lower": int(delta.lt(0).sum()),
    }


def plot(weekly, delta):
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    set_style()
    arms = {name: frame.set_index("friday") for name, frame in weekly.groupby("arm")}
    one, seven, zero = (arms[key] for key in (*LEARNED, "Zero"))
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.9), sharex=True)
    fig.suptitle(
        "One-year ridge forecasts · one versus seven channels",
        x=0.09,
        y=0.98,
        ha="left",
        fontsize=19,
        fontweight="bold",
    )
    fig.text(
        0.09,
        0.929,
        "Same training examples and same 15-stock comparison basket · 33 Fridays in 2026",
    )
    colors = {"One channel": "#7860aa", "Seven channels": "#126f73", "Zero": "#748396"}
    line(axes[0], one.basket_actual, "Actual basket", "#172b4d")
    line(axes[0], one.spy, "Realized SPY", "#c37224", style="--")
    for arm, frame in (("One channel", one), ("Seven channels", seven)):
        line(axes[0], frame.basket_forecast, arm, colors[arm])
    decorate(
        axes[0], "Forecast and realized return for the same 15 stocks", "Return (%)"
    )
    for arm, frame in (("Seven channels", seven), ("Zero", zero)):
        line(
            axes[1], frame.basket_mae_pp - one.basket_mae_pp, arm, colors[arm], scale=1
        )
    decorate(
        axes[1],
        f"Absolute error relative to one channel · seven minus one {delta['seven_minus_one_pp']:+.3f} pp on average",
        "Error difference (pp)",
    )
    fig.subplots_adjust(left=0.09, right=0.97, top=0.84, bottom=0.08, hspace=0.56)
    fig.savefig(FIGURE, dpi=150)
    plt.close(fig)


def main():
    RESULT.mkdir(parents=True, exist_ok=True)
    calendar, _, close, volume, logs, _ = session_data()
    anchors = decision_dates(logs.index, evaluation=False)
    label_ends = logs.index[logs.index.get_indexer(anchors) + 5]
    train = anchors[label_ends < VALIDATION]
    validation = anchors[
        (anchors >= VALIDATION) & (label_ends < pd.Timestamp("2026-01-01"))
    ]
    historical = daily_features(calendar, close, volume, logs, evaluation=False)
    models, settings, counts = fitted_models(logs, historical, train, validation)
    del historical
    current = daily_features(calendar, close, volume, logs, evaluation=True)
    scores = score(logs, current, decision_dates(logs.index, evaluation=True), models)
    panel, weekly = evaluation(scores)
    weekly.to_csv(RESULT / "weekly_forecasts.csv", index=False)
    with duckdb.connect() as db:
        db.from_df(panel.reset_index()).write_parquet(
            str(RESULT / "stock_forecasts.parquet")
        )
    weights = RESULT / "ridge_weights.npz"
    np.savez(
        weights,
        one_weights=models["One channel"][0],
        one_intercept=models["One channel"][1],
        seven_weights=models["Seven channels"][0],
        seven_intercept=models["Seven channels"][1],
    )
    primary = compare(weekly, "basket_mae_pp")
    paths = [
        ROSTER,
        DATA / "daily_ohlcv_2016_2025.csv",
        DATA / "friday/flagged_daily_returns.csv",
        DATA / "news/news_v1.duckdb",
        ROOT / "data/v1_holdout/prices_2026.csv",
        ROOT / "data/v1_holdout/news/news_v1.duckdb",
        FORECAST_EVIDENCE / "stock_forecasts.parquet",
        FORECAST_EVIDENCE / "weekly_forecasts.csv",
    ]
    repo = ROOT.parents[1]
    summary = {
        **counts,
        "fit_decisions": [str(train[0].date()), str(train[-1].date())],
        "validation_decisions": [str(validation[0].date()), str(validation[-1].date())],
        "test_weeks": int(weekly.friday.nunique()),
        "test_stock_outcomes": len(panel),
        "settings": settings,
        "baseline_metrics": json.loads(
            weekly.groupby("arm")[["basket_mae_pp", "stock_mae_pp", "rank_ic"]]
            .mean()
            .to_json(orient="index", double_precision=15)
        ),
        "basket_primary": primary,
        "stock_diagnostic": compare(weekly, "stock_mae_pp"),
        "saved_weights_sha256": sha256(weights),
        "source_sha256": {str(path.relative_to(repo)): sha256(path) for path in paths},
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=repo
        ).strip(),
        "runner_sha256": sha256(Path(__file__)),
        "news_loader_sha256": sha256(
            ROOT.parent / "AlphaAllocationV2/build_friday_dataset.py"
        ),
    }
    (RESULT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(weekly, primary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

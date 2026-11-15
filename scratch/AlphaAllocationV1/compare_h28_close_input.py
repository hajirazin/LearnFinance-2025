"""Compare H28 adjusted-close price and return inputs on identical replay rows."""

import json
import subprocess
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
from compare_patchtst import sha256
from compare_v1_forecast_accuracy import (
    decorate,
    line,
    paired_interval,
    plt,
    set_style,
)
from run_v1_h28 import FORECAST_EVIDENCE, load_model, predict
from train_v1_h28 import RESULT, RETURN_RESULT, replace_price_with_return
from train_v1_hypotheses import model_factory
from v1_hypothesis_data import daily_features, decision_dates, session_data, windows


def model_scores():
    calendar, _, close, volume, logs, _ = session_data()
    dates = decision_dates(close.index, evaluation=True)
    features = daily_features(calendar, close, volume, logs, evaluation=True)
    index, x, mask, _ = windows(logs, features, dates, use_features=True, year=False)
    price_meta = json.loads((RESULT / "h28.json").read_text())
    return_meta = json.loads((RETURN_RESULT / "h28.json").read_text())
    for field in (
        "policy",
        "seed",
        "fit_start",
        "fit_end",
        "fit_anchors",
        "train_rows",
        "training_symbols",
        "validation_rows",
        "config",
    ):
        if price_meta[field] != return_meta[field]:
            raise ValueError(f"Training metadata mismatch for {field}")
    weights = RETURN_RESULT / "h28.pt"
    if (
        sha256(weights) != return_meta["weights_sha256"]
        or return_meta["close_input"] != "return"
    ):
        raise ValueError("Wrong checkpoint for the return arm")
    price = load_model("H28", x, "cpu")
    challenger = model_factory(x, daily=True)
    challenger.load_state_dict(
        torch.load(weights, weights_only=True, map_location="cpu")
    )
    scores = pd.DataFrame(
        {
            "H28 price": predict(price, x, mask, "cpu"),
            "H28 return": predict(
                challenger.eval(), replace_price_with_return(x, mask), mask, "cpu"
            ),
        },
        index=index,
    )
    archived = pd.read_csv(RESULT / "scores.csv", parse_dates=["friday"])
    archived["friday"] = archived.friday.dt.as_unit("ns")
    archived = archived.set_index(["friday", "symbol"]).H28
    if not scores.index.equals(archived.index):
        raise ValueError("H28 inference roster changed")
    np.testing.assert_allclose(scores["H28 price"], archived, atol=5e-7, rtol=1e-5)
    return scores, price_meta, return_meta


def matched_panel(scores):
    with duckdb.connect() as db:
        panel = db.read_parquet(str(FORECAST_EVIDENCE / "stock_forecasts.parquet")).df()
    panel["friday"] = panel.friday.dt.as_unit("ns")
    panel = panel.set_index(["friday", "symbol"])[["actual"]].dropna()
    panel = panel.join(scores, how="left", validate="one_to_one")
    if panel.isna().any().any():
        raise ValueError("Missing forecasts for archived stock outcomes")
    panel["Zero"] = 0.0
    fixed = pd.read_csv(
        FORECAST_EVIDENCE / "weekly_forecasts.csv", parse_dates=["friday"]
    )
    fixed = fixed.loc[fixed.method.eq("Native") & fixed.model.eq("V1")].set_index(
        "friday"
    )
    fixed.index = fixed.index.as_unit("ns")
    if not panel.index.get_level_values("friday").unique().equals(fixed.index):
        raise ValueError("Matched dates differ from the archive")
    return panel, fixed


def weekly_metrics(panel, fixed):
    records = []
    for date, week in panel.groupby(level="friday", sort=True):
        picks = fixed.loc[date, "fixed_symbols"].split(",")
        if len(set(picks)) != 15:
            raise ValueError(f"Invalid fixed basket on {date}")
        basket = week.droplevel("friday").loc[picks]
        actual, realized_ranks = week.actual, week.actual.rank()
        for name in ("H28 price", "H28 return", "Zero"):
            errors = 100 * (week[name] - actual)
            basket_error = 100 * (basket[name] - basket.actual).mean()
            records.append(
                {
                    "friday": date,
                    "model": name,
                    "matched_stocks": len(week),
                    "stock_mae_pp": errors.abs().mean(),
                    "stock_mse_pp2": errors.pow(2).mean(),
                    "stock_bias_pp": errors.mean(),
                    "rank_ic": week[name].rank().corr(realized_ranks)
                    if name != "Zero"
                    else np.nan,
                    "basket_actual": basket.actual.mean(),
                    "basket_forecast": basket[name].mean(),
                    "basket_error_pp": basket_error,
                    "basket_mae_pp": abs(basket_error),
                    "spy": fixed.loc[date, "spy"],
                }
            )
    return pd.DataFrame(records)


def difference(weekly, field):
    paired = weekly.pivot(index="friday", columns="model", values=field)
    mean, low, high, p = paired_interval(paired["H28 return"] - paired["H28 price"])
    return {
        "return_minus_price_pp": mean,
        "ci95_pp": [low, high],
        "p_for_lower_error": p,
    }


def draw(weekly, primary):
    set_style()
    by_model = {
        name: group.set_index("friday") for name, group in weekly.groupby("model")
    }
    price, return_, zero = (
        by_model[name] for name in ("H28 price", "H28 return", "Zero")
    )
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 9.3), sharex=True)
    fig.suptitle(
        "H28 · Price history versus daily returns",
        x=0.09,
        y=0.982,
        ha="left",
        fontsize=20,
        fontweight="bold",
    )
    fig.text(
        0.09,
        0.935,
        "Same stocks and fixed 15-name basket each week · 33 matched Fridays in 2026",
        fontsize=10,
    )
    colors = {
        "Price input": "#7860aa",
        "Return input": "#126f73",
        "Actual": "#172b4d",
        "SPY": "#c37224",
        "Zero": "#748396",
    }
    line(axes[0], price.basket_actual, "Actual basket", colors["Actual"])
    line(axes[0], price.basket_forecast, "Price input", colors["Price input"])
    line(axes[0], return_.basket_forecast, "Return input", colors["Return input"])
    line(axes[0], price.spy, "Realized SPY", colors["SPY"], style="--")
    decorate(
        axes[0], "Forecasts and realized returns for the same 15 stocks", "Return (%)"
    )

    delta = (return_.basket_mae_pp - price.basket_mae_pp).asfreq("W-FRI")
    axes[1].bar(
        delta.index,
        delta,
        width=4.6,
        color=[
            colors["Return input"] if d < 0 else colors["Price input"] for d in delta
        ],
        label="Negative favors the daily-return input",
    )
    decorate(
        axes[1],
        f"Paired basket error difference · mean {primary['return_minus_price_pp']:+.2f} pp",
        "Return − price (pp)",
    )

    for frame, label in (
        (price, "Price input"),
        (return_, "Return input"),
        (zero, "Zero"),
    ):
        line(axes[2], frame.stock_mae_pp, label, colors[label], scale=1)
    decorate(
        axes[2],
        "Mean absolute error across matched individual stocks",
        "Error (pp)",
        zero=False,
    )
    axes[-1].set_xlim(pd.Timestamp("2026-01-04"), pd.Timestamp("2026-11-07"))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.860, bottom=0.06, hspace=0.64)
    fig.savefig(RETURN_RESULT / "forecast_comparison.png", dpi=155)
    plt.close(fig)


def main():
    torch.set_num_threads(4)
    scores, price, return_ = model_scores()
    panel, fixed = matched_panel(scores)
    weekly = weekly_metrics(panel, fixed)
    weekly.to_csv(RETURN_RESULT / "forecast_weekly.csv", index=False)
    with duckdb.connect() as db:
        db.from_df(panel.reset_index()).write_parquet(
            str(RETURN_RESULT / "matched_stock_forecasts.parquet")
        )
    metrics = weekly.groupby("model")[
        ["stock_mae_pp", "stock_mse_pp2", "stock_bias_pp", "basket_mae_pp", "rank_ic"]
    ].mean()
    metrics["stock_rmse_pp"] = metrics.pop("stock_mse_pp2").pow(0.5)
    metrics["basket_rmse_pp"] = (
        weekly.assign(squared_error=weekly.basket_error_pp.pow(2))
        .groupby("model")
        .squared_error.mean()
        .pow(0.5)
    )
    primary = difference(weekly, "basket_mae_pp")
    errors = weekly.pivot(index="friday", columns="model", values="basket_mae_pp")
    primary["weeks_return_input_had_lower_error"] = int(
        errors["H28 return"].lt(errors["H28 price"]).sum()
    )
    tracked_inputs = [
        FORECAST_EVIDENCE / "stock_forecasts.parquet",
        FORECAST_EVIDENCE / "weekly_forecasts.csv",
        RESULT / "training_coverage.csv",
        RETURN_RESULT / "training_coverage.csv",
        Path(__file__).with_name("V1_H28_CLOSE_RETURN.md"),
    ]
    repo = Path(__file__).resolve().parents[2]
    prior_sources = json.loads((RESULT / "summary.json").read_text())["sources_sha256"]
    for relative, digest in prior_sources.items():
        path = repo / relative
        if path.suffix == ".py":
            continue
        if sha256(path) != digest:
            raise ValueError(f"Archived H28 data changed: {relative}")
    summary = {
        "weeks": len(fixed),
        "stock_outcomes": len(panel),
        "first_decision": str(fixed.index.min().date()),
        "last_decision": str(fixed.index.max().date()),
        "training_rows": return_["train_rows"],
        "training_symbols": return_["training_symbols"],
        "validation_rows": return_["validation_rows"],
        "seed": return_["seed"],
        "saved_epochs": {
            "H28 price": price["best_epoch"],
            "H28 return": return_["best_epoch"],
        },
        "checkpoint_validation_weekly_rank_ic": {
            label: row["epochs"][row["best_epoch"] - 1]["validation_rank_ic"]
            for label, row in (("H28 price", price), ("H28 return", return_))
        },
        "weights_sha256": {
            "H28 price": price["weights_sha256"],
            "H28 return": return_["weights_sha256"],
        },
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "analysis_sha256": sha256(Path(__file__)),
        "inputs_sha256": {
            str(path.relative_to(repo)): sha256(path) for path in tracked_inputs
        },
        "metrics": json.loads(metrics.to_json(orient="index", double_precision=15)),
        "basket_mae_by_spy_direction_pp": (
            weekly.assign(
                spy_direction=np.where(weekly.spy.lt(0), "down", "up_or_flat")
            )
            .groupby(["spy_direction", "model"])
            .basket_mae_pp.mean()
            .unstack()
            .to_dict(orient="index")
        ),
        "basket_mae_primary": primary,
        "stock_mae_diagnostic": difference(weekly, "stock_mae_pp"),
    }
    (RETURN_RESULT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    draw(weekly, primary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

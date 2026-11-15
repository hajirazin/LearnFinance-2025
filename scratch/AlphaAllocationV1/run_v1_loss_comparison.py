"""Evaluate H21–H23 and frozen baselines on common 2026 candidates."""

import argparse

import numpy as np
import pandas as pd
import torch
from compare_feature_sets import forecast_accuracy
from compare_patchtst import sha256, write_json
from compare_v1_rank_market import RESULT as BENCHMARK_DATA
from compare_v1_rank_market import ranking_week
from run_v1_holdout import TERMINAL
from run_v1_hypotheses import ALPHA, BPS, holm, one_sided_test, scores_for
from train_v1_loss_comparison import RESULT
from v1_holdout_portfolio import allocation_path, simulate
from v1_hypothesis_data import (
    OUTPUT,
    daily_features,
    decision_dates,
    session_data,
    windows,
)

BASELINES = ("current", "h5_features_training_filter")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    calendar, prices, close, volume, logs, _ = session_data()
    features = daily_features(calendar, close, volume, logs, evaluation=True)
    dates = decision_dates(close.index, evaluation=True)
    index, x, mask, daily_actual = windows(
        logs, features, dates, use_features=True, year=False
    )
    baseline_file = OUTPUT / "scores.csv"
    baseline = pd.read_csv(baseline_file, parse_dates=["friday"]).set_index(
        ["friday", "symbol"]
    )
    if not baseline.index.is_unique:
        raise ValueError("Duplicate saved baseline predictions")
    scores = baseline.loc[index, list(BASELINES)].copy()
    for model in ("h21", "h22", "h23"):
        scores[model] = scores_for(
            model, x, mask, args.device, checkpoint_dir=RESULT
        ).astype(np.float64)
    scores = scores.dropna()
    if scores.groupby(level="friday").size().reindex(dates, fill_value=0).lt(15).any():
        raise ValueError("Missing common evaluation candidates")
    actual = pd.Series(
        np.expm1(daily_actual.sum(axis=1, dtype=np.float64)),
        index=index,
        name="actual_five_session_return",
    ).reindex(scores.index)
    stock_scores = scores.join(actual)
    keys = scores.index.to_frame(index=False)
    exact_friday_returns = (
        close.div(
            close.reindex(close.index - pd.Timedelta(days=7)).set_axis(close.index)
        )
        - 1
    )
    friday_outcomes = exact_friday_returns.stack().rename_axis(["end_friday", "symbol"])
    target_keys = pd.MultiIndex.from_arrays(
        [keys["friday"] + pd.Timedelta(days=7), keys["symbol"]]
    )
    stock_scores["actual_friday_return"] = friday_outcomes.reindex(
        target_keys
    ).to_numpy()
    stock_scores["actual_rank"] = (
        stock_scores["actual_friday_return"]
        .sort_index()
        .groupby(level="friday")
        .rank(ascending=False, method="first")
    )
    for name in scores:
        stock_scores[f"{name}_rank"] = (
            scores[name]
            .sort_index()
            .groupby(level="friday")
            .rank(ascending=False, method="first")
        )
    stock_scores.to_csv(RESULT / "stock_scores.csv")

    baskets, ranking, ledger, nav, picks = [], [], [], [], []
    portfolios = {}
    for model in scores:
        path = allocation_path(scores[model], prices["close"], equal_weight=True)
        for friday, weights, selection in path:
            predicted = scores.loc[friday, model]
            outcome = actual.loc[friday]
            basket = weights.index
            picks.extend(
                {
                    "arm": model,
                    "friday": friday,
                    "symbol": symbol,
                    "weight": weight,
                    "reason": selection.reasons[symbol],
                }
                for symbol, weight in weights.items()
            )
            # H22 and H23 output scores optimized for selection. Only the
            # return-trained arms' values are interpreted as return forecasts.
            if model in (*BASELINES, "h21"):
                baskets.append(
                    {
                        "arm": model,
                        "friday": friday,
                        "predicted": predicted.loc[basket].mean(),
                        "actual": outcome.loc[basket].mean(skipna=False),
                    }
                )
            next_friday = friday + pd.Timedelta(days=7)
            if next_friday in close.index:
                friday_returns = close.loc[next_friday] / close.loc[friday] - 1
                record = ranking_week(predicted, friday_returns, basket)
                if record is not None:
                    ranking.append({"arm": model, "friday": friday, **record})
        result, weekly, daily, _ = simulate(
            path, prices["open"], prices["raw_open"], prices["close"], TERMINAL, BPS
        )
        portfolios[model] = result
        ledger.append(weekly.assign(arm=model))
        nav.append(daily.assign(arm=model))

    basket_frame, ranking_frame = map(pd.DataFrame, (baskets, ranking))
    basket_wide = basket_frame.pivot(
        index="friday", columns="arm", values="actual"
    ).dropna()
    hits = ranking_frame.pivot(
        index="friday", columns="arm", values="sticky_top15_hits"
    ).dropna()
    forecast_metrics = {}
    for model in (*BASELINES, "h21"):
        basket = basket_frame.loc[
            basket_frame["arm"].eq(model)
            & basket_frame["friday"].isin(basket_wide.index)
        ]
        forecast_metrics[model] = {
            "stock": forecast_accuracy(scores[model], actual),
            "basket": forecast_accuracy(basket["predicted"], basket["actual"]),
        }

    weekly = pd.concat(ledger, ignore_index=True)
    spy = pd.read_csv(
        BENCHMARK_DATA / "spy_adjusted_prices.csv", parse_dates=["date"]
    ).set_index("date")["open"]
    weekly["spy_return"] = weekly["next_mark"].map(spy) / weekly["fill"].map(spy) - 1
    if not np.isfinite(weekly["spy_return"]).all():
        raise ValueError("Missing SPY prices on evaluation dates")
    weekly["spy_excess"] = weekly["net_return"] - weekly["spy_return"]
    market = {}
    for model, rows in weekly.groupby("arm"):
        down = rows["spy_return"].lt(0)
        market[model] = {
            "spy_total_return": (1 + rows["spy_return"]).prod() - 1,
            "periods_ahead": int(rows["spy_excess"].gt(0).sum()),
            "down_periods": int(down.sum()),
            "down_periods_ahead": int(rows.loc[down, "spy_excess"].gt(0).sum()),
            "mean_down_excess": rows.loc[down, "spy_excess"].mean(),
        }
    errors = (
        basket_frame.assign(
            squared_error=(basket_frame["predicted"] - basket_frame["actual"]) ** 2
        )
        .pivot(index="friday", columns="arm", values="squared_error")
        .dropna()
    )
    h23 = weekly.loc[weekly["arm"].eq("h23")]
    duration = (h23["next_mark"] - h23["fill"]).dt.days / 7
    spy_excess = np.expm1(np.log1p(h23["net_return"]) / duration) - np.expm1(
        np.log1p(h23["spy_return"]) / duration
    )
    diffs = {
        "h21": errors["current"] - errors["h21"],
        "h22": hits["h22"] - hits["current"],
        "h23": spy_excess,
    }
    tests = {
        name: dict(
            zip(("p_raw", "ci95"), one_sided_test(diff), strict=True),
            mean_effect=diff.mean(),
            weeks=len(diff),
        )
        for name, diff in diffs.items()
    }
    for name, p in holm({name: row["p_raw"] for name, row in tests.items()}).items():
        tests[name].update(p_holm=p, reject_null=p < ALPHA)

    for name, frame in {
        "basket_forecasts": basket_frame,
        "rank_weekly": ranking_frame,
        "portfolio_weekly": weekly,
        "picks": pd.DataFrame(picks),
        "nav": pd.concat(nav),
    }.items():
        frame.to_csv(RESULT / f"{name}.csv", index=False)
    columns = [
        "rank_ic",
        "strict_top15_hits",
        "sticky_top15_hits",
        "random_expected_hits",
    ]
    write_json(
        RESULT / "summary.json",
        {
            "baseline_scores_sha256": sha256(baseline_file),
            "training_spy_sha256": sha256(RESULT / "spy_training_closes.csv"),
            "evaluation_spy_sha256": sha256(BENCHMARK_DATA / "spy_adjusted_prices.csv"),
            "checkpoints_sha256": {
                model: sha256(RESULT / f"{model}.pt") for model in ("h21", "h22", "h23")
            },
            "periods": len(dates),
            "rank_weeks": len(hits),
            "basket_forecast_weeks": len(basket_wide),
            "forecast_accuracy": forecast_metrics,
            "ranking": ranking_frame.loc[ranking_frame["friday"].isin(hits.index)]
            .groupby("arm")[columns]
            .mean()
            .to_dict("index"),
            "portfolio": portfolios,
            "market": market,
            "tests": tests,
        },
    )


if __name__ == "__main__":
    main()

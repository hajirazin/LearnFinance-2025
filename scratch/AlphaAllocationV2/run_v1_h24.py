"""Compare H24 and frozen controls with SAC's IBKR Singapore fees."""

import argparse

import numpy as np
import pandas as pd
import torch
from compare_feature_sets import forecast_accuracy
from compare_patchtst import sha256, write_json
from compare_v1_rank_market import RESULT as BENCHMARK_DATA
from compare_v1_rank_market import ranking_week
from run_v1_holdout import TERMINAL
from run_v1_hypotheses import ALPHA, BPS, one_sided_test, scores_for
from train_v1_loss_comparison import RESULT as LOSS_RESULT
from v1_holdout_portfolio import allocation_path, simulate
from v1_hypothesis_data import daily_features, decision_dates, session_data, windows
from v1_ibkr_cost_loss import COST_CONFIG, COST_WEIGHT, ibkr_transaction_fees

RESULT = LOSS_RESULT / "h24_ibkr"
MODELS = ("current", "h21", "h21_chronological", "h24")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    calendar, prices, close, volume, logs, _ = session_data()
    dates = decision_dates(close.index, evaluation=True)
    features = daily_features(calendar, close, volume, logs, evaluation=True)
    index, x, mask, _ = windows(logs, features, dates, use_features=True, year=False)
    source = LOSS_RESULT / "stock_scores.csv"
    scores = pd.read_csv(source, parse_dates=["friday"]).set_index(["friday", "symbol"])
    scores = scores[
        ["current", "h21", "actual_five_session_return", "actual_friday_return"]
    ].copy()
    if not scores.index.is_unique or set(scores.index) != set(index):
        raise ValueError("H24 candidates must match the saved H21 replay")
    for model in ("h21_chronological", "h24"):
        scores[model] = pd.Series(
            scores_for(model, x, mask, args.device, checkpoint_dir=RESULT),
            index=index,
            dtype=float,
        )
    scores.to_csv(RESULT / "stock_scores.csv")
    spy = pd.read_csv(
        BENCHMARK_DATA / "spy_adjusted_prices.csv", parse_dates=["date"]
    ).set_index("date")["open"]

    summaries, ledgers, navs, orders, baskets, ranks = {}, [], [], [], [], []
    for model in MODELS:
        print(f"Replaying {model} under SAC IBKR fees", flush=True)
        path = allocation_path(scores[model], prices["close"], equal_weight=True)
        for day, weights, selection in path:
            week = scores.loc[day]
            chosen = weights.index
            baskets.append(
                {
                    "arm": model,
                    "friday": day,
                    "predicted": week.loc[chosen, model].mean(),
                    "actual": week.loc[chosen, "actual_five_session_return"].mean(
                        skipna=False
                    ),
                    "kept": selection.kept_count,
                    "symbols": ",".join(chosen),
                }
            )
            if week["actual_friday_return"].notna().any():
                metrics = ranking_week(
                    week[model], week["actual_friday_return"], chosen
                )
                if metrics is not None:
                    ranks.append({"arm": model, "friday": day, **metrics})
        result, weekly, nav, trades = simulate(
            path,
            prices["open"],
            prices["raw_open"],
            prices["close"],
            TERMINAL,
            BPS,
            fee_model=ibkr_transaction_fees,
        )
        weekly = weekly.rename(columns={"regulatory_fees": "ibkr_fees"})
        weekly["spy_return"] = (
            weekly["next_mark"].map(spy) / weekly["fill"].map(spy) - 1
        )
        if not np.isfinite(weekly["spy_return"]).all():
            raise ValueError("Missing SPY returns")
        weekly["spy_excess"] = weekly["net_return"] - weekly["spy_return"]
        summaries[model] = {
            **result,
            "ibkr_fees_usd": weekly["ibkr_fees"].sum(),
            "spread_slippage_usd": weekly["spread_slippage"].sum(),
            "spy_total_return": (1 + weekly["spy_return"]).prod() - 1,
            "periods_beating_spy": int(weekly["spy_excess"].gt(0).sum()),
            "stock_forecast": forecast_accuracy(
                scores[model], scores["actual_five_session_return"]
            ),
        }
        ledgers.append(weekly.assign(arm=model))
        navs.append(nav.assign(arm=model))
        orders.append(trades.assign(arm=model))

    ledger = pd.concat(ledgers, ignore_index=True)
    basket, ranking = pd.DataFrame(baskets), pd.DataFrame(ranks)
    matched_basket = (
        basket.pivot(index="friday", columns="arm", values="actual").dropna().index
    )
    matched_rank = (
        ranking.pivot(index="friday", columns="arm", values="sticky_top15_hits")
        .dropna()
        .index
    )
    for model, summary in summaries.items():
        complete = basket.loc[
            basket["arm"].eq(model) & basket["friday"].isin(matched_basket)
        ]
        ranked = ranking.loc[
            ranking["arm"].eq(model) & ranking["friday"].isin(matched_rank)
        ]
        summary["basket_forecast"] = forecast_accuracy(
            complete["predicted"], complete["actual"]
        )
        summary["sticky_top15_hits"] = ranked["sticky_top15_hits"].mean()
        summary["rank_ic"] = ranked["rank_ic"].mean()
        summary["mean_names_replaced"] = (
            15 - basket.loc[basket["arm"].eq(model), "kept"].iloc[1:]
        ).mean()
    ledger["weekly_equivalent_return"] = np.expm1(
        np.log1p(ledger["net_return"])
        / ((ledger["next_mark"] - ledger["fill"]).dt.days / 7)
    )
    paired = ledger.pivot(
        index="friday", columns="arm", values="weekly_equivalent_return"
    )
    delta = paired["h24"] - paired["h21"]
    p, ci = one_sided_test(delta)
    isolated = paired["h24"] - paired["h21_chronological"]
    isolated_p, isolated_ci = one_sided_test(isolated)
    for name, frame in {
        "portfolio_weekly": ledger,
        "nav": pd.concat(navs),
        "trades": pd.concat(orders),
        "basket_forecasts": basket,
        "rank_weekly": ranking,
    }.items():
        frame.to_csv(RESULT / f"{name}.csv", index=False)
    write_json(
        RESULT / "summary.json",
        {
            "baseline_scores_sha256": sha256(source),
            "h24_weights_sha256": sha256(RESULT / "h24.pt"),
            "chronological_control_sha256": sha256(RESULT / "h21_chronological.pt"),
            "cost_weight": COST_WEIGHT,
            "ibkr_rates": COST_CONFIG.to_dict(),
            "spread_slippage_bps_per_side": BPS,
            "periods": len(dates),
            "rank_weeks": len(matched_rank),
            "arms": summaries,
            "h24_vs_h21": {
                "p_one_sided": p,
                "reject_null": p < ALPHA,
                "ci95": ci,
                "mean_weekly_net_difference": delta.mean(),
                "weeks_ahead": int(delta.gt(0).sum()),
            },
            "h24_vs_chronological_control": {
                "p_one_sided": isolated_p,
                "reject_null": isolated_p < ALPHA,
                "ci95": isolated_ci,
                "mean_weekly_net_difference": isolated.mean(),
                "weeks_ahead": int(isolated.gt(0).sum()),
            },
        },
    )


if __name__ == "__main__":
    main()

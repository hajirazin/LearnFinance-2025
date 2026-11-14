"""Replay H25 and V1 with equal weights and HRP on identical dates."""

import argparse
import json
from itertools import pairwise

import numpy as np
import pandas as pd
import torch
from brain_api.core.sticky_selection import select_with_rank_band
from build_friday_dataset import plt
from compare_patchtst import sha256, write_json
from compare_v1_rank_market import RESULT as BENCHMARK_DATA
from compare_v1_rank_market import ranking_week
from matplotlib.ticker import PercentFormatter
from run_v1_holdout import TERMINAL
from run_v1_hypotheses import BPS, one_sided_test
from train_v1_h25 import RESULT, RankPatchTST
from train_v1_loss_comparison import RESULT as LOSS_RESULT
from v1_holdout_portfolio import INITIAL_NAV, allocation_path, simulate
from v1_hypothesis_data import daily_features, decision_dates, session_data, windows
from v1_ibkr_cost_loss import ibkr_transaction_fees

LABELS = {
    "v1": "V1 close-only",
    "h25": "H25 rank-only",
    "oracle": "Actual top 15 · hindsight",
    "spy": "SPY",
}
COLORS = {"v1": "#35688e", "h25": "#a4522a", "oracle": "#5f6873", "spy": "#548564"}


def rank_scores(index, x, mask, model, device):
    """Run the selector once, passing only its prior selections to the model."""
    previous, values = set(), []
    dates = index.get_level_values("friday")
    with torch.inference_mode():
        for date in dates.unique().sort_values():
            rows = np.flatnonzero(dates == date)
            symbols = index[rows].get_level_values("symbol")
            selected_before = torch.tensor(
                [s in previous for s in symbols], dtype=torch.float32, device=device
            )
            prediction = (
                model(
                    torch.tensor(x[rows], device=device),
                    torch.tensor(mask[rows], device=device),
                    selected_before,
                )
                .cpu()
                .numpy()
            )
            score = pd.Series(prediction, index=index[rows], dtype=float)
            selection = select_with_rank_band(
                dict(zip(symbols, prediction.tolist(), strict=True)), None, 15, 15
            )
            previous = set(selection.selected)
            values.append(score)
    return pd.concat(values).reindex(index)


def oracle_scores(scores, dates, prices):
    opens, raw, closes = (prices[name] for name in ("open", "raw_open", "close"))
    fills = opens.index[opens.index.searchsorted(dates, side="right")]
    records = []
    for day, start, end in zip(dates, fills, [*fills[1:], TERMINAL], strict=True):
        universe = scores.loc[day].index
        returns = opens.loc[end, universe] / opens.loc[start, universe] - 1
        observed = raw.loc[start, universe].gt(0) & raw.loc[end, universe].gt(0)
        observed &= closes.loc[start:end, universe].gt(0).all(axis=0)
        week = returns.where(observed & np.isfinite(returns)).dropna()
        if len(week) < 15:
            raise ValueError(
                f"Insufficient realized returns for the hindsight basket on {day}"
            )
        records.append(
            week.rename_axis("symbol")
            .rename("oracle")
            .to_frame()
            .assign(friday=day)
            .reset_index()
            .set_index(["friday", "symbol"])
        )
    return pd.concat(records)["oracle"]


def chart(weekly, weighting):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.0), sharex=True, layout="constrained")
    for name, rows in weekly.groupby("arm", sort=False):
        days = pd.DatetimeIndex([rows["fill"].iloc[0], *rows["next_mark"]])
        growth = np.r_[1, (1 + rows["net_return"]).cumprod().to_numpy()]
        axes[0].plot(days, growth, label=LABELS[name], color=COLORS[name], linewidth=2)
        if name != "oracle":
            axes[1].plot(days, growth - 1, color=COLORS[name], linewidth=2)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Growth of $1 (log scale)")
    axes[0].set_title("All four portfolios", loc="left", fontsize=11)
    axes[0].legend(frameon=False, ncol=2)
    axes[1].set_ylabel("Cumulative net return")
    axes[1].set_title("V1, H25, and SPY", loc="left", fontsize=11)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].axhline(0, color="#888888", linewidth=0.8)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    title = (
        "Equal-weight top 15" if weighting == "equal_weight" else "HRP-weighted top 15"
    )
    fig.suptitle(f"{title} · January–November 2026")
    fig.savefig(RESULT / f"{weighting}.png", dpi=155)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    calendar, prices, close, volume, logs, _ = session_data()
    dates = decision_dates(close.index, evaluation=True)
    features = daily_features(calendar, close, volume, logs, evaluation=True)
    index, x, mask, _ = windows(logs, features, dates, use_features=True, year=False)
    saved = pd.read_csv(
        LOSS_RESULT / "stock_scores.csv", parse_dates=["friday"]
    ).set_index(["friday", "symbol"])
    if not saved.index.is_unique or set(index) != set(saved.index):
        raise ValueError("H25 candidates must match the saved V1 comparison")
    model = RankPatchTST(x.shape[1]).to(args.device).eval()
    checkpoint = RESULT / "h25.pt"
    metadata = json.loads((RESULT / "h25.json").read_text())
    if sha256(checkpoint) != metadata["weights_sha256"]:
        raise ValueError("H25 checkpoint hash mismatch")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    scores = (
        saved[["current", "actual_friday_return", "actual_rank"]]
        .rename(columns={"current": "v1"})
        .copy()
    )
    scores["h25"] = rank_scores(index, x, mask, model, args.device)
    scores["v1_rank"] = saved["current_rank"]
    scores["h25_rank"] = (
        scores["h25"]
        .sort_index()
        .groupby(level="friday")
        .rank(ascending=False, method="first")
    )
    scores["oracle_open_return"] = oracle_scores(scores, dates, prices)
    scores.to_csv(RESULT / "stock_ranks.csv")

    rank_rows = []
    for name in ("v1", "h25"):
        for day, weights, _ in allocation_path(
            scores[name], prices["close"], equal_weight=True, sticky=name == "v1"
        ):
            week = scores.loc[day]
            if week["actual_friday_return"].notna().any():
                row = ranking_week(
                    week[name], week["actual_friday_return"], weights.index
                )
                if row is not None:
                    rank_rows.append({"arm": name, "friday": day, **row})
    rankings = pd.DataFrame(rank_rows).rename(
        columns={
            "sticky_top15_hits": "selected_top15_hits",
            "sticky_hits_symbols": "selected_hits_symbols",
            "sticky_realized_rank_percentile": "selected_realized_rank_percentile",
            "sticky_return": "selected_friday_return",
            "oracle_return": "actual_top15_friday_return",
        }
    )
    rankings.to_csv(RESULT / "rank_weekly.csv", index=False)
    hits = rankings.pivot(
        index="friday", columns="arm", values="selected_top15_hits"
    ).dropna()
    rank_summary = (
        rankings.loc[rankings["friday"].isin(hits.index)]
        .groupby("arm")[
            [
                "rank_ic",
                "strict_top15_hits",
                "selected_top15_hits",
                "random_expected_hits",
            ]
        ]
        .mean()
        .to_dict("index")
    )
    p, ci = one_sided_test(hits["h25"] - hits["v1"])
    rank_test = {"weeks": len(hits), "p_one_sided": p, "ci95_hits": ci}

    spy = pd.read_csv(
        BENCHMARK_DATA / "spy_adjusted_prices.csv", parse_dates=["date"]
    ).set_index("date")["open"]
    result, weights_rows, ledgers, navs = {}, [], [], []
    for weighting in ("equal_weight", "hrp"):
        result[weighting] = {}
        weekly = []
        for name, signal in (
            ("v1", scores["v1"]),
            ("h25", scores["h25"]),
            ("oracle", scores["oracle_open_return"].dropna()),
        ):
            path = allocation_path(
                signal,
                prices["close"],
                equal_weight=weighting == "equal_weight",
                sticky=name == "v1",
            )
            for day, weights, _ in path:
                weights_rows.extend(
                    {
                        "weighting": weighting,
                        "arm": name,
                        "friday": day,
                        "symbol": symbol,
                        "weight": value,
                    }
                    for symbol, value in weights.items()
                )
            metrics, ledger, nav, _ = simulate(
                path,
                prices["open"],
                prices["raw_open"],
                prices["close"],
                TERMINAL,
                BPS,
                fee_model=ibkr_transaction_fees,
            )
            weekly.append(ledger.assign(arm=name))
            navs.append(nav.assign(weighting=weighting, arm=name))
            result[weighting][name] = metrics
        marks = ledger[["friday", "fill", "next_mark"]].copy()
        marks["net_return"] = marks["next_mark"].map(spy) / marks["fill"].map(spy) - 1
        if not np.isfinite(marks["net_return"]).all():
            raise ValueError("Missing SPY execution opens")
        weekly.append(marks.assign(arm="spy"))
        result[weighting]["spy"] = {"net_return": (1 + marks["net_return"]).prod() - 1}
        combined = pd.concat(weekly, ignore_index=True).assign(weighting=weighting)
        period_returns = combined.pivot(
            index="fill", columns="arm", values="net_return"
        )
        for name in ("v1", "h25"):
            excess = period_returns[name] - period_returns["spy"]
            down = period_returns["spy"].lt(0)
            result[weighting][name].update(
                periods_ahead_of_spy=int(excess.gt(0).sum()),
                spy_down_periods=int(down.sum()),
                down_periods_ahead_of_spy=int((down & excess.gt(0)).sum()),
                mean_excess_in_spy_down_periods=excess[down].mean(),
            )
        chart(combined, weighting)
        ledgers.append(combined)
    holdings = pd.DataFrame(weights_rows)
    selections = (
        holdings.groupby(["weighting", "arm", "friday"])["symbol"]
        .agg(frozenset)
        .unstack("weighting")
    )
    if not selections["equal_weight"].eq(selections["hrp"]).all():
        raise ValueError("The weighting methods must use identical selected names")
    for name in ("v1", "h25", "oracle"):
        baskets = selections["equal_weight"].xs(name).sort_index().tolist()
        replaced = np.mean(
            [len(current - prior) for prior, current in pairwise(baskets)]
        )
        for weighting in result:
            result[weighting][name]["mean_names_replaced"] = replaced
    holdings.to_csv(RESULT / "picks_and_weights.csv", index=False)
    pd.concat(ledgers).to_csv(RESULT / "portfolio_weekly.csv", index=False)
    pd.concat(navs).to_csv(RESULT / "nav.csv", index=False)
    write_json(
        RESULT / "summary.json",
        {
            "periods": len(dates),
            "initial_nav": INITIAL_NAV,
            "rank": rank_summary,
            "rank_test": rank_test,
            "previous_top15_score_coefficient": model.head.weight[0, -1].item(),
            "portfolio": result,
            "checkpoint_sha256": sha256(checkpoint),
        },
    )


if __name__ == "__main__":
    main()

"""Compare seven-target forecasts with V1, H5, and H21."""

import argparse
import json

import numpy as np
import pandas as pd
import torch
from build_friday_dataset import plt
from compare_feature_sets import forecast_accuracy
from compare_patchtst import sha256, write_json
from compare_v1_rank_market import RESULT as BENCHMARK_DATA
from compare_v1_rank_market import ranking_week
from matplotlib.ticker import PercentFormatter
from run_v1_holdout import TERMINAL
from run_v1_hypotheses import BPS, holm, one_sided_test
from train_v1_loss_comparison import RESULT as LOSS_RESULT
from train_v1_multitarget import ARMS, RESULT, MultiTargetPatchTST, close_forecast
from v1_holdout_portfolio import allocation_path, simulate
from v1_hypothesis_data import daily_features, decision_dates, session_data, windows
from v1_ibkr_cost_loss import ibkr_transaction_fees

BASELINES = {"current": "v1", "h5_features_training_filter": "h5", "h21": "h21"}
NAMES = ("v1", "h5", "h21", *ARMS)
COLORS = {
    "v1": "#375e8c",
    "h5": "#807e83",
    "h21": "#82649f",
    "h26": "#b45229",
    "h27": "#338278",
    "spy": "#5c703a",
}


def predict(name, x, mask, device):
    weights = RESULT / f"{name}.pt"
    meta = json.loads((RESULT / f"{name}.json").read_text())
    if sha256(weights) != meta["weights_sha256"] or meta["horizon"] != ARMS[name]:
        raise ValueError(f"Checkpoint mismatch for {name}")
    model = MultiTargetPatchTST(x.shape[1], meta["horizon"]).to(device).eval()
    model.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"))
    with torch.inference_mode():
        output = torch.cat(
            [
                close_forecast(
                    model(
                        torch.as_tensor(x[i : i + 256], device=device),
                        torch.as_tensor(mask[i : i + 256], device=device),
                    ),
                    daily=True,
                    mean=meta["means"][0],
                    scale=meta["scales"][0],
                ).cpu()
                for i in range(0, len(x), 256)
            ]
        ).numpy()
    if not np.isfinite(output).all():
        raise ValueError(f"Nonfinite forecasts for {name}")
    return output, meta


def compare_ranks(scores, selections, sessions):
    rank_rows = []
    for day, week in scores.groupby(level="friday"):
        if not week["actual_friday_return"].notna().any():
            continue
        week = week.droplevel("friday")
        records = []
        for name in NAMES:
            picks = selections[name][day]
            row = ranking_week(week[name], week["actual_friday_return"], picks)
            if row is not None:
                records.append(
                    {
                        "friday": day,
                        "arm": name,
                        **row,
                        "predicted_basket_return": week.loc[list(picks), name].mean(),
                    }
                )
        if len(records) == len(NAMES):
            rank_rows.extend(records)
    ranks = pd.DataFrame(rank_rows)
    ranks.to_csv(RESULT / "rank_weekly.csv", index=False)
    summary = {}
    rank_dates = pd.DatetimeIndex(ranks.friday.unique())
    # Match both target horizons for forecast errors. Keep shorter Friday pairs
    # in the ordering and basket-return metrics.
    five_session_dates = rank_dates[
        sessions.get_indexer(rank_dates + pd.Timedelta(days=7))
        - sessions.get_indexer(rank_dates)
        == 5
    ]
    matched = scores.loc[
        scores.index.get_level_values("friday").isin(five_session_dates)
    ]
    for name, week in ranks.groupby("arm"):
        forecast_weeks = week.loc[week.friday.isin(five_session_dates)]
        summary[name] = {
            "rank_ic": week.rank_ic.mean(),
            "strict_top15_hits": week.strict_top15_hits.mean(),
            "selected_top15_hits": week.sticky_top15_hits.mean(),
            "mean_selected_friday_return": week.sticky_return.mean(),
            "stock_forecast": forecast_accuracy(
                matched[name], matched.actual_friday_return
            ),
            "basket_forecast": forecast_accuracy(
                forecast_weeks.predicted_basket_return, forecast_weeks.sticky_return
            ),
        }
    return ranks, summary


def plot_portfolios(weekly):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout="constrained")
    for ax, (weighting, title) in zip(
        axes, (("equal_weight", "Equal weight"), ("hrp", "HRP")), strict=True
    ):
        for name, rows in weekly.loc[weekly.weighting.eq(weighting)].groupby("arm"):
            dates = pd.DatetimeIndex([rows.fill.iloc[0], *rows.next_mark])
            growth = np.r_[1, (1 + rows.net_return).cumprod()]
            ax.plot(
                dates,
                growth - 1,
                color=COLORS[name],
                label=name.upper(),
                linewidth=2,
                linestyle="--" if name == "spy" else "-",
            )
        ax.axhline(0, color="#777777", linewidth=0.8)
        ax.grid(axis="y", alpha=0.25)
        ax.set_title(title, loc="left")
        ax.set_ylabel("Cumulative net return")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(ncol=6, frameon=False)
    fig.suptitle("Seven-target PatchTST vs. prior models · 2026")
    fig.savefig(RESULT / "portfolio_comparison.png", dpi=160)
    plt.close(fig)


def replay(scores, prices):
    spy = pd.read_csv(
        BENCHMARK_DATA / "spy_adjusted_prices.csv", parse_dates=["date"]
    ).set_index("date")["open"]
    selections, metrics, ledgers, holdings = {}, {}, [], []
    for weighting in ("equal_weight", "hrp"):
        metrics[weighting] = {}
        for name in NAMES:
            path = allocation_path(
                scores[name], prices["close"], equal_weight=weighting == "equal_weight"
            )
            baskets = {day: frozenset(weights.index) for day, weights, _ in path}
            if weighting == "equal_weight":
                selections[name] = baskets
            elif baskets != selections[name]:
                raise ValueError(f"Allocators selected different names for {name}")
            for day, weights, _ in path:
                holdings.extend(
                    {
                        "friday": day,
                        "weighting": weighting,
                        "arm": name,
                        "symbol": symbol,
                        "weight": value,
                    }
                    for symbol, value in weights.items()
                )
            result, ledger, _, _ = simulate(
                path,
                prices["open"],
                prices["raw_open"],
                prices["close"],
                TERMINAL,
                BPS,
                fee_model=ibkr_transaction_fees,
            )
            ledger["spy_return"] = ledger.next_mark.map(spy) / ledger.fill.map(spy) - 1
            if not np.isfinite(ledger.spy_return).all():
                raise ValueError("Incomplete SPY execution prices")
            result["periods_ahead_of_spy"] = int(
                (ledger.net_return > ledger.spy_return).sum()
            )
            result["net_spy_return"] = float((1 + ledger.spy_return).prod() - 1)
            metrics[weighting][name] = result
            ledgers.append(ledger.assign(weighting=weighting, arm=name))
        ledgers.append(
            ledger[["friday", "fill", "next_mark", "spy_return"]]
            .rename(columns={"spy_return": "net_return"})
            .assign(weighting=weighting, arm="spy")
        )
    weekly = pd.concat(ledgers, ignore_index=True)
    weekly.to_csv(RESULT / "portfolio_weekly.csv", index=False)
    pd.DataFrame(holdings).to_csv(RESULT / "picks_and_weights.csv", index=False)
    plot_portfolios(weekly)
    return selections, metrics, weekly


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    calendar, prices, close, volume, logs, _ = session_data()
    dates = decision_dates(close.index, evaluation=True)
    features = daily_features(calendar, close, volume, logs, evaluation=True)
    index, x, mask, _ = windows(logs, features, dates, use_features=True, year=False)
    saved_file = LOSS_RESULT / "stock_scores.csv"
    saved = pd.read_csv(saved_file, parse_dates=["friday"]).set_index(
        ["friday", "symbol"]
    )
    if not saved.index.is_unique or set(index) != set(saved.index):
        raise ValueError("Candidate set differs from the archived replay")
    scores = saved.loc[
        index, [*BASELINES, "actual_friday_return", "actual_rank"]
    ].rename(columns=BASELINES)
    checkpoints = {}
    for name in ARMS:
        scores[name], checkpoints[name] = predict(name, x, mask, args.device)
    for name in NAMES:
        scores[f"{name}_rank"] = (
            scores[name]
            .sort_index()
            .groupby(level="friday")
            .rank(ascending=False, method="first")
        )
    scores.to_csv(RESULT / "stock_scores.csv")
    selections, portfolios, weekly = replay(scores, prices)
    ranks, rank_metrics = compare_ranks(scores, selections, close.index)
    hrp = weekly.loc[weekly.weighting.eq("hrp")]
    returns = hrp.pivot(index="friday", columns="arm", values="net_return")
    duration = hrp.drop_duplicates("friday").set_index("friday")
    days = (duration.next_mark - duration.fill).dt.days
    normalized = np.expm1(np.log1p(returns).mul(7 / days, axis=0))
    tests = {}
    for name in ARMS:
        diff = normalized[name] - normalized["v1"]
        p, ci = one_sided_test(diff)
        tests[name] = {"weekly_mean_difference": diff.mean(), "p_raw": p, "ci95": ci}
    for name, adjusted in holm(
        {name: row["p_raw"] for name, row in tests.items()}
    ).items():
        tests[name]["p_holm"] = adjusted
    controls = {}
    for name, reference in (("h26", "h5"), ("h27", "h21")):
        diff = normalized[name] - normalized[reference]
        _, ci = one_sided_test(diff)
        controls[name] = {
            "reference": reference,
            "weekly_mean_difference": diff.mean(),
            "ci95": ci,
        }
    write_json(
        RESULT / "summary.json",
        {
            "periods": len(dates),
            "rank_weeks": ranks.friday.nunique(),
            "candidate_count_range": [
                int(scores.groupby(level="friday").size().min()),
                int(scores.groupby(level="friday").size().max()),
            ],
            "ranking": rank_metrics,
            "portfolio": portfolios,
            "hrp_tests_vs_v1": tests,
            "hrp_differences_vs_close_only_targets": controls,
            "checkpoints_sha256": {
                name: meta["weights_sha256"] for name, meta in checkpoints.items()
            },
            "baseline_scores_sha256": sha256(saved_file),
        },
    )


if __name__ == "__main__":
    main()

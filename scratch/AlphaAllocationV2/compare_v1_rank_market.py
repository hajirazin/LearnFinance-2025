"""Compare frozen V1 forecasts with realized ranks and market benchmarks."""

import json

import numpy as np
import pandas as pd
from compare_patchtst import sha256, write_json
from run_v1_hypotheses import ALPHA, holm, one_sided_test
from v1_hypothesis_data import ARMS, OUTPUT

SOURCE = OUTPUT / "equal_weight"
RESULT = OUTPUT / "diagnostics"
PRICES = OUTPUT.parent / "v1_holdout/prices_2026.csv"
TOP_K = 15


def ranking_week(scores, actual, picks):
    """Find the realized top 15, with ticker order breaking return ties."""
    available = actual.reindex(scores.index).dropna().sort_index()
    if len(available) < TOP_K:
        raise ValueError("Insufficient realized returns for the top 15")
    oracle = available.sort_values(ascending=False, kind="stable").head(TOP_K)
    strict = scores.sort_index().sort_values(ascending=False, kind="stable").head(TOP_K)
    observed_picks = available.reindex(picks).dropna()
    if len(observed_picks) != TOP_K or available.reindex(strict.index).isna().any():
        return None
    selected = set(picks)
    hits = len(selected & set(oracle.index))
    return {
        "candidates_with_outcomes": len(available),
        "rank_ic": scores.loc[available.index].rank().corr(available.rank()),
        "strict_top15_hits": len(set(strict.index) & set(oracle.index)),
        "sticky_top15_hits": hits,
        "random_expected_hits": TOP_K**2 / len(available),
        "sticky_realized_rank_percentile": (
            1
            - (available.rank(ascending=False).loc[list(picks)] - 1)
            / (len(available) - 1)
        ).mean(),
        "sticky_return": observed_picks.mean(),
        "oracle_return": oracle.mean(),
        "actual_top15": ",".join(oracle.index),
        "sticky_hits_symbols": ",".join(s for s in oracle.index if s in selected),
    }


def rank_comparisons(scores, closes, weights, periods):
    by_model, records = {}, []
    for number, model in enumerate(list(ARMS)[1:], start=11):
        pair = scores[["current", model]].dropna()
        selections = (
            weights.loc[weights["comparison"].eq(model)]
            .groupby(["friday", "arm"])["symbol"]
            .agg(list)
        )
        weeks = []
        for friday in periods:
            end = friday + pd.Timedelta(days=7)
            if end not in closes.index:
                continue
            actual = (closes.loc[end] / closes.loc[friday] - 1).replace(
                [np.inf, -np.inf], np.nan
            )
            predictions = pair.loc[friday]
            metrics = {
                arm: ranking_week(
                    predictions[arm], actual, selections.loc[(friday, arm)]
                )
                for arm in pair
            }
            if any(row is None for row in metrics.values()):
                continue
            weeks.extend(
                {
                    "hypothesis": f"H{number}",
                    "comparison": model,
                    "arm": arm,
                    "friday": friday,
                    "end_friday": end,
                    **row,
                }
                for arm, row in metrics.items()
            )
        records.extend(weeks)
        frame = pd.DataFrame(weeks)
        values = frame.pivot(index="friday", columns="arm", values="sticky_top15_hits")
        difference = values[model] - values["current"]
        p, ci = one_sided_test(difference)
        columns = [
            "rank_ic",
            "strict_top15_hits",
            "sticky_top15_hits",
            "sticky_realized_rank_percentile",
            "sticky_return",
            "oracle_return",
            "random_expected_hits",
        ]
        by_model[f"H{number}"] = {
            "model": model,
            "weeks": len(values),
            "mean_sticky_hit_difference": difference.mean(),
            "ci95": ci,
            "p_raw": p,
            "arms": frame.groupby("arm")[columns].mean().to_dict("index"),
        }
    return by_model, pd.DataFrame(records)


def market_comparisons(weekly, benchmarks):
    by_model, records = {}, []
    for number, model in enumerate(list(ARMS)[1:], start=16):
        ledger = weekly.loc[weekly["comparison"].eq(model)].copy()
        for benchmark, prices in benchmarks.items():
            entry = ledger["fill"].map(prices)
            exit_ = ledger["next_mark"].map(prices)
            if not (
                np.isfinite(entry).all()
                and np.isfinite(exit_).all()
                and entry.gt(0).all()
                and exit_.gt(0).all()
            ):
                raise ValueError(f"Missing or invalid benchmark prices for {benchmark}")
            ledger[f"{benchmark}_return"] = exit_ / entry - 1
            ledger[f"{benchmark}_excess"] = (
                ledger["net_return"] - ledger[f"{benchmark}_return"]
            )
        ledger["hypothesis"] = f"H{number}"
        records.append(ledger)
        metrics = {}
        for arm, rows in ledger.groupby("arm"):
            metrics[arm] = {}
            duration = (rows["next_mark"] - rows["fill"]).dt.days / 7
            for benchmark in benchmarks:
                market = rows[f"{benchmark}_return"]
                excess = rows[f"{benchmark}_excess"]
                portfolio = rows["net_return"]
                weekly_excess = np.expm1(np.log1p(portfolio) / duration) - np.expm1(
                    np.log1p(market) / duration
                )
                p, ci = one_sided_test(weekly_excess)
                down = market.lt(0)
                metrics[arm][benchmark] = {
                    "periods": len(rows),
                    "portfolio_total_return": (1 + portfolio).prod() - 1,
                    "market_total_return": (1 + market).prod() - 1,
                    "compounded_return_difference": (1 + portfolio).prod()
                    - (1 + market).prod(),
                    "periods_beating_market": int(excess.gt(0).sum()),
                    "market_down_periods": int(down.sum()),
                    "down_periods_beating_market": int((down & excess.gt(0)).sum()),
                    "down_period_mean_excess": excess[down].mean(),
                    "mean_weekly_excess": weekly_excess.mean(),
                    "ci95": ci,
                    "p_raw": p,
                }
        by_model[f"H{number}"] = {
            "model": model,
            "arms": metrics,
            "p_raw": metrics[model]["SPY"]["p_raw"],
        }
    return by_model, pd.concat(records, ignore_index=True)


def main():
    score_file = OUTPUT / "scores.csv"
    previous = json.loads((SOURCE / "summary.json").read_text())
    if previous["scores_sha256"] != sha256(score_file):
        raise ValueError("Saved forecasts differ from the equal-weight replay")
    scores = pd.read_csv(score_file, parse_dates=["friday"]).set_index(
        ["friday", "symbol"]
    )
    weights = pd.read_csv(SOURCE / "weights.csv", parse_dates=["friday"])
    weekly = pd.read_csv(
        SOURCE / "weekly_results.csv", parse_dates=["friday", "fill", "next_mark"]
    )
    prices = pd.read_csv(PRICES, parse_dates=["date"])
    closes = prices.pivot(index="date", columns="symbol", values="close").where(
        lambda values: values.gt(0)
    )
    spy_path = RESULT / "spy_adjusted_prices.csv"
    spy = pd.read_csv(spy_path, parse_dates=["date"]).set_index("date")["open"]
    spus = prices.loc[prices["symbol"].eq("SPUS")].set_index("date")["open"]
    ranks, rank_weekly = rank_comparisons(
        scores, closes, weights, scores.index.unique("friday").sort_values()
    )
    markets, market_weekly = market_comparisons(weekly, {"SPY": spy, "SPUS": spus})
    results = {**ranks, **markets}
    adjusted = holm({key: value["p_raw"] for key, value in results.items()})
    for key, value in results.items():
        value.update(p_holm=adjusted[key], reject_null=adjusted[key] < ALPHA)
    rank_weekly.to_csv(RESULT / "ranking_weekly.csv", index=False)
    market_weekly.to_csv(RESULT / "market_weekly.csv", index=False)
    write_json(
        RESULT / "summary.json",
        {
            "alpha": ALPHA,
            "holm_family_size": len(results),
            "sources_sha256": {
                path.name: sha256(path)
                for path in [
                    score_file,
                    PRICES,
                    spy_path,
                    SOURCE / "weights.csv",
                    SOURCE / "weekly_results.csv",
                ]
            },
            "rank_hypotheses": ranks,
            "market_hypotheses": markets,
        },
    )


if __name__ == "__main__":
    main()

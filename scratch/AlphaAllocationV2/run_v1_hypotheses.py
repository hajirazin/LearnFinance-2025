"""Evaluate five paired V1 hypotheses with common costs and portfolio weights."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from compare_patchtst import sha256, write_json
from patchtst_replay_models import model_output, simple_returns
from run_v1_holdout import TERMINAL
from train_v1_hypotheses import model_factory
from v1_holdout_portfolio import allocation_path, simulate
from v1_hypothesis_data import (
    ARMS,
    OUTPUT,
    daily_features,
    decision_dates,
    session_data,
    windows,
)

ALPHA = 0.05
BLOCK = 4
REPS = 20_000
BPS = 10


def one_sided_test(differences):
    """Approximate a zero-mean null by resampling centered circular blocks."""
    values = np.asarray(differences, dtype=float)
    if len(values) <= BLOCK or not np.isfinite(values).all():
        raise ValueError("Insufficient valid paired observations")
    observed = values.mean()
    rng = np.random.default_rng(42)
    starts = rng.integers(
        0, len(values), size=(REPS, (len(values) + BLOCK - 1) // BLOCK)
    )
    idx = ((starts[:, :, None] + np.arange(BLOCK)) % len(values)).reshape(REPS, -1)[
        :, : len(values)
    ]
    bootstrap = values[idx].mean(axis=1)
    p = (1 + np.count_nonzero(bootstrap - observed >= observed)) / (REPS + 1)
    return float(p), np.quantile(bootstrap, [0.025, 0.975]).tolist()


def holm(pvalues):
    adjusted = {}
    floor = 0.0
    for i, (name, p) in enumerate(sorted(pvalues.items(), key=lambda pair: pair[1])):
        floor = min(1.0, max(floor, (len(pvalues) - i) * p))
        adjusted[name] = floor
    return adjusted


def scores_for(name, x, mask, device="cpu"):
    metadata = json.loads((OUTPUT / f"{name}.json").read_text())
    weights = OUTPUT / f"{name}.pt"
    if sha256(weights) != metadata["weights_sha256"]:
        raise ValueError(f"Mismatched weights for {name}")
    model = model_factory(x, daily=True).eval()
    model.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"))
    model.to(device)
    with torch.inference_mode():
        x = torch.from_numpy(x)
        mask = torch.from_numpy(mask)
        result = (
            torch.cat(
                [
                    simple_returns(
                        model_output(
                            model, values.to(device), observed.to(device), daily=True
                        ),
                        daily=True,
                    )
                    for values, observed in zip(
                        x.split(256), mask.split(256), strict=True
                    )
                ]
            )
            .cpu()
            .numpy()
        )
    if not np.isfinite(result).all():
        raise ValueError(f"Nonfinite predictions for {name}")
    return result


def compare_all(scores, dates, prices, *, equal_weight=False):
    """Compare all arms on one candidate set for the portfolio leaderboard."""
    common = scores.dropna()
    candidates = common.groupby(level="friday").size().reindex(dates, fill_value=0)
    if candidates.lt(15).any():
        raise ValueError("Fewer than 15 candidates shared by all models")
    results = {}
    for arm in common:
        path = allocation_path(common[arm], prices["close"], equal_weight=equal_weight)
        results[arm], *_ = simulate(
            path, prices["open"], prices["raw_open"], prices["close"], TERMINAL, BPS
        )
    return {
        "candidates_min": int(candidates.min()),
        "candidates_max": int(candidates.max()),
        "portfolios": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument(
        "--equal-weight",
        action="store_true",
        help="Replay the saved forecasts at 1/15 weights.",
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    calendar, prices, close, volume, logs, smaller = session_data()
    dates = decision_dates(close.index, evaluation=True)
    score_path = OUTPUT / "scores.csv"
    if args.equal_weight:
        output = OUTPUT / "equal_weight"
        scores = pd.read_csv(score_path, parse_dates=["friday"]).set_index(
            ["friday", "symbol"]
        )
        scores = scores[list(ARMS)]
        if not scores.index.is_unique or set(scores.index.unique("friday")) != set(
            dates
        ):
            raise ValueError(
                "Saved forecasts must cover the exact evaluation dates with unique keys"
            )
    else:
        output = OUTPUT
        features = daily_features(calendar, close, volume, logs, evaluation=True)
        forecasts = []
        for name, (seven, year, _restricted) in ARMS.items():
            index, x, mask, _ = windows(
                logs, features, dates, use_features=seven, year=year
            )
            forecasts.append(
                pd.Series(
                    scores_for(name, x, mask, args.device), index=index, name=name
                )
            )
        scores = pd.concat(forecasts, axis=1)
        scores.to_csv(score_path)
    summary, ledgers, weights, navs, trades = {}, [], [], [], []
    for name in list(ARMS)[1:]:
        pair = scores[["current", name]].dropna()
        candidates = pair.groupby(level="friday").size().reindex(dates, fill_value=0)
        if candidates.lt(15).any():
            raise ValueError(
                f"Missing comparable inference coverage for {name}: {candidates[candidates.lt(15)]}"
            )
        paired_weekly = {}
        portfolios = {}
        for arm in pair:
            print(f"Simulating {name} / {arm} over {len(dates)} decisions", flush=True)
            path = allocation_path(
                pair[arm], prices["close"], equal_weight=args.equal_weight
            )
            for friday, allocation, selection in path:
                weights.extend(
                    {
                        "comparison": name,
                        "arm": arm,
                        "friday": friday,
                        "symbol": symbol,
                        "weight": weight,
                        "selection_reason": selection.reasons[symbol],
                    }
                    for symbol, weight in allocation.items()
                )
            result, ledger, nav, orders = simulate(
                path, prices["open"], prices["raw_open"], prices["close"], TERMINAL, BPS
            )
            ledgers.append(ledger.assign(comparison=name, arm=arm))
            navs.append(nav.assign(comparison=name, arm=arm))
            trades.append(orders.assign(comparison=name, arm=arm))
            portfolios[arm] = result
            weeks = (ledger["next_mark"] - ledger["fill"]).dt.days / 7
            paired_weekly[arm] = np.expm1(np.log1p(ledger["net_return"]) / weeks)
        diff = paired_weekly[name] - paired_weekly["current"]
        p, ci = one_sided_test(diff)
        inferred_symbols = pair.index.get_level_values("symbol")
        changed = set(inferred_symbols) - smaller
        summary[name] = {
            "periods": len(diff),
            "candidates_min": int(candidates.min()),
            "candidates_max": int(candidates.max()),
            "inference_symbols": int(inferred_symbols.nunique()),
            "inference_symbols_outside_smaller_cohort": len(changed),
            "selected_symbols_outside_smaller_cohort": len(
                {row["symbol"] for row in weights if row["arm"] == name} & changed
            ),
            "portfolios": portfolios,
            "mean_paired_weekly_net_difference": diff.mean(),
            "ci95_weekly": ci,
            "compounded_net_difference": portfolios[name]["net_return"]
            - portfolios["current"]["net_return"],
            "periods_ahead": int(diff.gt(0).sum()),
            "p_raw": p,
        }
    adjusted = holm({name: row["p_raw"] for name, row in summary.items()})
    for name, result in summary.items():
        result.update(p_holm=adjusted[name], reject_null=adjusted[name] < ALPHA)
    output.mkdir(exist_ok=True)
    pd.concat(ledgers).to_csv(output / "weekly_results.csv", index=False)
    pd.concat(navs).to_csv(output / "nav.csv", index=False)
    pd.concat(trades).to_csv(output / "trades.csv", index=False)
    pd.DataFrame(weights).to_csv(output / "weights.csv", index=False)
    write_json(
        output / "summary.json",
        {
            "alpha": ALPHA,
            "weighting": "equal_weight" if args.equal_weight else "hrp",
            "scores_sha256": sha256(score_path),
            "basis_points_per_side": BPS,
            "bootstrap": {"block": BLOCK, "replicates": REPS, "seed": 42},
            "first_decision": dates[0],
            "last_decision": dates[-1],
            "terminal_open": TERMINAL,
            "checkpoint_sha256": {name: sha256(OUTPUT / f"{name}.pt") for name in ARMS},
            "all_arms_common_universe": compare_all(
                scores, dates, prices, equal_weight=args.equal_weight
            ),
            "protocol_sha256": sha256(Path(__file__).with_name("V1_HYPOTHESES.md")),
            "comparisons": summary,
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

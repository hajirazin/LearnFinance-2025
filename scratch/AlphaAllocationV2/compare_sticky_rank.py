"""Replay strict and production 15/20 rank-band baskets on saved 2025 scores."""

import csv
import hashlib
import json
import random
from pathlib import Path
from statistics import mean, median, quantiles

from brain_api.core.sticky_selection import rank_by_score, select_with_rank_band

SOURCE = Path(__file__).resolve().parent / "data/friday/patchtst_comparison"
MODELS = ("ridge", "weekly_patchtst", "daily_patchtst")
TOP_N = 15
HOLD_THRESHOLD = 20


def load_forecasts(files):
    forecasts = {}
    for file in files:
        with file.open(newline="") as handle:
            for row in csv.DictReader(handle):
                week = forecasts.setdefault(row["friday"], {})
                symbol = row["symbol"]
                if symbol in week:
                    raise ValueError(f"Duplicate forecast for {row['friday']} {symbol}")
                actual = float(row["forward_return"]) if row["forward_return"] else None
                week[symbol] = {
                    **{model: float(row[model]) for model in MODELS},
                    "actual": actual,
                }
    return forecasts


def replay(forecasts, *, top_n=TOP_N, hold_threshold=HOLD_THRESHOLD):
    previous = {}
    weekly, picks = [], []
    for friday, rows in sorted(forecasts.items()):
        actual = {
            symbol: row["actual"]
            for symbol, row in rows.items()
            if row["actual"] is not None
        }
        realized_ranks = (
            {symbol: rank for symbol, rank, _ in rank_by_score(actual)}
            if actual
            else {}
        )
        realized_top = {
            symbol for symbol, rank in realized_ranks.items() if rank <= top_n
        }
        week = []
        for model in MODELS:
            scores = {symbol: row[model] for symbol, row in rows.items()}
            if len(scores) < top_n:
                raise ValueError(f"Too few candidates on {friday}")
            ranks = {symbol: rank for symbol, rank, _ in rank_by_score(scores)}
            for policy in ("strict", "sticky"):
                prior = previous.get((model, policy))
                carry = prior if policy == "sticky" else None
                selection = select_with_rank_band(scores, carry, top_n, hold_threshold)
                selected = set(selection.selected)
                previous[model, policy] = selected
                missing = len(selected - actual.keys())
                week.append(
                    {
                        "friday": friday,
                        "model": model,
                        "selector": policy,
                        "candidates": len(scores),
                        "universe_return": mean(actual.values()) if actual else None,
                        "predicted_return": mean(scores[s] for s in selected),
                        "actual_return": None
                        if missing
                        else mean(actual[s] for s in selected),
                        "missing_outcomes": missing,
                        "realized_top15_hits": None
                        if missing
                        else len(selected & realized_top),
                        "fraction_replaced": None
                        if prior is None
                        else len(selected - prior) / top_n,
                    }
                )
                picks.extend(
                    {
                        "friday": friday,
                        "model": model,
                        "selector": policy,
                        "symbol": symbol,
                        "predicted_rank": ranks[symbol],
                        "predicted_return": scores[symbol],
                        "realized_rank": realized_ranks.get(symbol),
                        "actual_return": actual.get(symbol),
                        "selection_reason": selection.reasons[symbol],
                    }
                    for symbol in selection.selected
                )
        complete = all(row["actual_return"] is not None for row in week)
        weekly.extend({**row, "matched": complete} for row in week)
    return weekly, picks


def paired_interval(differences):
    block = 4
    if len(differences) < block:
        raise ValueError("At least four paired weeks are needed for the interval")
    rng = random.Random(42)
    means = []
    for _ in range(5000):
        starts = [
            rng.randrange(len(differences) - block + 1)
            for _ in range((len(differences) + block - 1) // block)
        ]
        resampled = [
            value for start in starts for value in differences[start : start + block]
        ]
        means.append(mean(resampled[: len(differences)]))
    bounds = quantiles(means, n=40, method="inclusive")
    return [bounds[0], bounds[-1]]


def summarize(weekly, *, matched):
    by_model = {}
    for model in MODELS:
        rows = [r for r in weekly if r["model"] == model]
        pairs = [
            (left, right)
            for left, right in zip(rows[::2], rows[1::2], strict=True)
            if (
                left["matched"]
                if matched
                else left["actual_return"] is not None
                and right["actual_return"] is not None
            )
        ]
        diff = [right["actual_return"] - left["actual_return"] for left, right in pairs]
        by_model[model] = {
            "weeks": len(pairs),
            "fridays": [left["friday"] for left, _ in pairs],
            "mean_sticky_minus_strict": mean(diff),
            "median_absolute_weekly_difference": median(map(abs, diff)),
            "mean_absolute_weekly_difference": mean(map(abs, diff)),
            "sticky_wins": sum(value > 0 for value in diff),
            "ci95": paired_interval(diff),
        }
        for i, policy in enumerate(("strict", "sticky")):
            baskets = [pair[i] for pair in pairs]
            turnover = [
                r["fraction_replaced"]
                for r in rows
                if r["selector"] == policy and r["fraction_replaced"] is not None
            ]
            by_model[model][policy] = {
                "mean_predicted_return": mean(r["predicted_return"] for r in baskets),
                "mean_actual_return": mean(r["actual_return"] for r in baskets),
                "basket_mae": mean(
                    abs(r["predicted_return"] - r["actual_return"]) for r in baskets
                ),
                "mean_top15_hits": mean(r["realized_top15_hits"] for r in baskets),
                "mean_fraction_replaced_all_scored_transitions": mean(turnover),
            }
    return by_model


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    source_bytes = (SOURCE / "summary.json").read_bytes()
    source_summary = json.loads(source_bytes)
    files = []
    for refit in source_summary["refits"]:
        file = SOURCE / "refits" / refit["refit"].split()[0] / "forecasts.csv"
        if hashlib.sha256(file.read_bytes()).hexdigest() != refit["forecasts_sha256"]:
            raise ValueError(
                f"Saved forecasts do not match the source manifest for {file}"
            )
        files.append(file)
    forecasts = load_forecasts(files)
    if len(forecasts) != source_summary["scored_weeks"]:
        raise ValueError("Saved forecast Fridays do not match the source manifest")
    weekly, picks = replay(forecasts)
    by_model = summarize(weekly, matched=True)
    matched_fridays = by_model[MODELS[0]]["fridays"]
    common = set(matched_fridays)
    sticky_weekly = {
        model: {
            r["friday"]: r["actual_return"]
            for r in weekly
            if r["model"] == model
            and r["selector"] == "sticky"
            and r["friday"] in common
        }
        for model in MODELS
    }
    differences = [
        sticky_weekly["weekly_patchtst"][week] - sticky_weekly["daily_patchtst"][week]
        for week in matched_fridays
    ]
    summary = {
        "source_summary_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "entry": TOP_N,
        "hold": HOLD_THRESHOLD,
        "scored_weeks": len(forecasts),
        "common_matched_weeks": len(matched_fridays),
        "common_matched_by_model": by_model,
        "matched_within_each_model": summarize(weekly, matched=False),
        "sticky_weekly_minus_daily": {
            "mean": mean(differences),
            "median_absolute_weekly_gap": median(map(abs, differences)),
            "ci95": paired_interval(differences),
        },
    }
    output = SOURCE / "sticky_rank"
    output.mkdir(exist_ok=True)
    write_csv(output / "weekly_results.csv", weekly)
    write_csv(output / "top_15_picks.csv", picks)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()

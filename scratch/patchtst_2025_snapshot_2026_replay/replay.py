"""Replay the fixed 2025 snapshot through completed 2026 weeks, without training."""

import argparse
import json
import logging
import pickle
import subprocess
from datetime import UTC, date, datetime

import duckdb
import numpy as np
import pandas as pd
import torch
from accounting import (
    completed_weekly_schedule,
    equal_weight_rebalance,
    mark_to_market,
    positive_price,
)
from prepare import DATA, ROOT, SNAPSHOT, VERSION, sha256, write_json

from brain_api.core.patchtst.calendar import load_patchtst_calendar
from brain_api.core.patchtst.weekly_data import WeeklyPatchTSTInputs
from brain_api.core.patchtst.weekly_training import predict_weekly_returns
from brain_api.core.sticky_selection import select_with_rank_band
from brain_api.storage.patchtst.local import PatchTSTHalalNewModelStorage

INITIAL = 10000.0
BENCHMARKS = {"SPY": "SPY", "QQQ": "QQQ", "S&P 500": "^GSPC"}


class ReadOnlyNewsStore:
    """Supply the production sentiment reader a connection that cannot write."""

    def _connect(self):
        return duckdb.connect(str(DATA / "news/news_v1.duckdb"), read_only=True)


def load_prices():
    manifest = json.loads((ROOT / "price_manifest.json").read_text())
    result = {}
    for name, expected in manifest["prices_sha256"].items():
        path = ROOT / "data/prices" / name
        if sha256(path) != expected:
            raise ValueError(f"Price evidence changed: {name}")
        frame = pd.read_csv(path, index_col="date", parse_dates=True)
        if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
            raise ValueError(f"Unordered or duplicate prices: {name}")
        result[path.stem] = frame
    return result


def prepare_inputs(artifacts, prices, calendar):
    path = ROOT / "data/inputs.pkl"
    identity = {
        "price_manifest_sha256": sha256(ROOT / "price_manifest.json"),
        "config_sha256": sha256(SNAPSHOT / "config.json"),
        "feature_code_sha256": sha256(
            DATA.parent / "brain_api/core/patchtst/weekly_data.py"
        ),
        "news_code_sha256": sha256(DATA.parent / "brain_api/news/patchtst.py"),
    }
    identity_path = ROOT / "data/inputs_identity.json"
    if path.exists() and identity_path.exists():
        previous = json.loads(identity_path.read_text())
        if previous["identity"] == identity and previous["sha256"] == sha256(path):
            with path.open("rb") as handle:
                return pickle.load(handle)
        raise ValueError("Existing feature evidence has a different identity")
    print(
        "Building causal inputs using the production feature/news implementation",
        flush=True,
    )
    store = ReadOnlyNewsStore()
    with store._connect() as con:
        news_evidence = con.execute(
            "SELECT min(window_start_exclusive), max(window_end_inclusive), count(*) FROM news_coverage"
        ).fetchone()
    inputs = WeeklyPatchTSTInputs(artifacts.config, calendar, prices, store)
    with path.open("wb") as handle:
        pickle.dump(inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    write_json(
        identity_path,
        {
            "identity": identity,
            "sha256": sha256(path),
            "news_coverage_inventory": news_evidence,
        },
    )
    return inputs


def score_weeks(inputs, artifacts, plan, roster):
    forecast_rows, exclusions = [], []
    for week in plan:
        windows, usable = [], []
        anchor = week["anchor"].date()
        for symbol in sorted(roster):
            window, count = inputs.window(symbol, anchor)
            if window is None:
                exclusions.append(
                    {
                        "monday": week["monday"],
                        "symbol": symbol,
                        "context_sessions": count,
                        "reason": "Production window rejected incomplete price/news/features",
                    }
                )
            else:
                windows.append(window)
                usable.append(symbol)
        if len(usable) < 15:
            raise ValueError(f"Only {len(usable)} eligible stocks on {anchor}")
        scores = predict_weekly_returns(
            artifacts.model, np.stack(windows), artifacts.feature_scaler
        )
        rank = {
            s: i + 1
            for i, s in enumerate(
                sorted(usable, key=lambda s: (-float(scores[usable.index(s)]), s))
            )
        }
        forecast_rows.extend(
            {
                "monday": week["monday"],
                "anchor": week["anchor"],
                "symbol": symbol,
                "predicted_return": float(score),
                "rank": rank[symbol],
                "model_target_end": inputs.calendar.forecast_end(anchor),
            }
            for symbol, score in zip(usable, scores, strict=True)
        )
        print(
            f"{week['monday'].date()}: scored {len(usable)} / {len(roster)}", flush=True
        )
    forecasts = pd.DataFrame(forecast_rows)
    forecasts.to_csv(ROOT / "forecasts.csv", index=False)
    pd.DataFrame(exclusions).to_csv(ROOT / "exclusions.csv", index=False)
    return forecasts


def quotes(prices, symbols, stamp, field):
    result = {}
    for symbol in symbols:
        frame = prices.get(symbol)
        if frame is None or stamp not in frame.index:
            raise ValueError(
                f"Missing execution/valuation price {symbol} {stamp} {field}"
            )
        result[symbol] = positive_price({symbol: frame.at[stamp, field]}, symbol)
    return result


def simulate(prices, forecasts, plan):
    shares, previous = {}, None
    rows = [
        {
            "date": plan[0]["execution"],
            "PatchTST": INITIAL,
            **dict.fromkeys(BENCHMARKS, INITIAL),
        }
    ]
    benchmark_quantity = {
        label: INITIAL / quotes(prices, [symbol], plan[0]["execution"], "open")[symbol]
        for label, symbol in BENCHMARKS.items()
    }
    ledger, holdings, trades = [], [], []
    previous_friday_nav = INITIAL
    for week in plan:
        monday = week["monday"]
        scores = (
            forecasts.loc[forecasts["monday"].eq(monday)]
            .set_index("symbol")["predicted_return"]
            .to_dict()
        )
        selection = select_with_rank_band(scores, previous, 15, 20)
        if len(selection.selected) != 15:
            raise ValueError("Sticky selection did not produce exactly 15 stocks")
        opening = quotes(
            prices, set(shares) | set(selection.selected), week["execution"], "open"
        )
        shares, rebalance = equal_weight_rebalance(
            shares, selection.selected, opening, INITIAL if previous is None else 0
        )
        closing = quotes(prices, shares, week["valuation"], "close")
        nav = mark_to_market(shares, closing)
        row = {"date": week["valuation"], "PatchTST": nav}
        for label, symbol in BENCHMARKS.items():
            row[label] = (
                benchmark_quantity[label]
                * quotes(prices, [symbol], week["valuation"], "close")[symbol]
            )
        rows.append(row)
        ledger.append(
            {
                **week,
                "prior_friday_nav": previous_friday_nav,
                "monday_open_nav": rebalance["nav_before"],
                "friday_close_nav": nav,
                "weekend_return": rebalance["nav_before"] / previous_friday_nav - 1,
                "open_to_close_return": nav / rebalance["nav_before"] - 1,
                "weekly_return": nav / previous_friday_nav - 1,
                "eligible_stocks": len(scores),
                "kept_count": selection.kept_count,
                "entered_count": selection.fillers_count,
                "selected": selection.selected,
                "evicted": selection.evicted_from_previous,
                "rebalance": rebalance,
            }
        )
        for symbol, quantity in shares.items():
            holdings.append(
                {
                    "monday": monday,
                    "execution": week["execution"],
                    "valuation": week["valuation"],
                    "symbol": symbol,
                    "quantity_adjusted_units": quantity,
                    "opening_price": opening[symbol],
                    "closing_price": closing[symbol],
                    "opening_value": quantity * opening[symbol],
                    "closing_value": quantity * closing[symbol],
                    "closing_weight": quantity * closing[symbol] / nav,
                    "predicted_return": scores[symbol],
                    "selection_reason": selection.reasons[symbol],
                }
            )
        trades.extend(
            {"monday": monday, "execution": week["execution"], **trade}
            for trade in rebalance["trades"]
        )
        previous, previous_friday_nav = set(shares), nav
    weekly = pd.DataFrame(rows).set_index("date")
    weekly.to_csv(ROOT / "weekly_nav.csv")
    pd.DataFrame(holdings).to_csv(ROOT / "weekly_holdings.csv", index=False)
    pd.DataFrame(holdings).loc[lambda f: f["monday"].eq(plan[-1]["monday"])].to_csv(
        ROOT / "ending_holdings.csv", index=False
    )
    pd.DataFrame(trades).to_csv(ROOT / "trades.csv", index=False)
    write_json(ROOT / "weekly_ledger.json", ledger)
    pd.DataFrame(
        [
            {
                k: v
                for k, v in week.items()
                if k not in {"rebalance", "evicted", "selected"}
            }
            | {"selected": ", ".join(week["selected"])}
            for week in ledger
        ]
    ).to_csv(ROOT / "weekly_portfolio.csv", index=False)
    return weekly, ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-inputs-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    metadata = json.loads((SNAPSHOT / "metadata.json").read_text())
    if metadata["training_evidence"]["latest_label_end"] >= "2026-01-01":
        raise ValueError("Snapshot includes 2026 labels")
    manifest = json.loads((ROOT / "price_manifest.json").read_text())
    for name, digest in manifest["snapshot_sha256"].items():
        if sha256(SNAPSHOT / name) != digest:
            raise ValueError(f"Snapshot changed: {name}")
    calendar = load_patchtst_calendar("us", date(2023, 11, 1), date(2026, 12, 10))
    plan = completed_weekly_schedule(
        calendar.closes, date(2026, 1, 5), date(2026, 11, 28), manifest["retrieved_at"]
    )
    artifacts = PatchTSTHalalNewModelStorage(DATA).load_version_artifacts(VERSION)
    prices = load_prices()
    inputs = prepare_inputs(
        artifacts, {s: prices[s] for s in metadata["symbols"] if s in prices}, calendar
    )
    if args.build_inputs_only:
        return
    forecasts = score_weeks(inputs, artifacts, plan, metadata["symbols"])
    weekly, ledger = simulate(prices, forecasts, plan)
    summary = {
        "snapshot": VERSION,
        "snapshot_trained_at": metadata["training_timestamp"],
        "snapshot_latest_label": metadata["training_evidence"]["latest_label_end"],
        "run_at": datetime.now(UTC),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "weeks": len(plan),
        "completed_weeks": sum(bool(row["week_complete"]) for row in plan),
        "final_week_complete": bool(plan[-1]["week_complete"]),
        "price_evidence_as_of": manifest["retrieved_at"],
        "roster_size": len(metadata["symbols"]),
        "initial_capital": INITIAL,
        "first_execution": plan[0]["execution"],
        "last_valuation": plan[-1]["valuation"],
        "top_n": 15,
        "hold_rank": 20,
        "transaction_cost": 0,
        "weekend": "Hold positions; self-financing rebalance at next session open",
        "metrics": {
            label: {
                "ending_value": float(weekly[label].iloc[-1]),
                "total_return": float(weekly[label].iloc[-1] / INITIAL - 1),
                "max_weekly_drawdown": float(
                    (weekly[label] / weekly[label].cummax() - 1).min()
                ),
            }
            for label in weekly.columns
        },
        "min_weekly_eligible": min(row["eligible_stocks"] for row in ledger),
        "max_weekly_eligible": max(row["eligible_stocks"] for row in ledger),
        "mean_retained": float(np.mean([r["kept_count"] for r in ledger[1:]])),
        "limitations": [
            "Retrospective frozen current roster; survivorship bias, not point-in-time membership.",
            "Snapshot fitted in November 2026 using labels ending December 2025; retrospective out-of-time experiment, not an artifact actually available in January 2026.",
            "Adjusted Yahoo prices model corporate actions/dividend reinvestment; quantities are adjusted units, not historical broker share counts.",
            "SPY and QQQ use adjusted-price returns; S&P 500 (^GSPC) is a price index without dividends.",
            "Model predicts five exchange sessions. Holiday-shortened weeks can target the following week rather than that Friday; no model alteration.",
            "Archived sentiment rejects unavailable future article revisions and uses production missingness rules; residual archive coverage is reported.",
            "Zero fees, spread, slippage, taxes, and market impact; fractional units; weekly-close drawdown only.",
        ],
        "code_sha256": {p.name: sha256(p) for p in ROOT.glob("*.py")},
    }
    write_json(ROOT / "summary.json", summary)
    print(json.dumps(summary["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()

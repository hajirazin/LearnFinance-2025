"""Reuse the completed snapshot replay and add only the requested HRP arm."""

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from hrp_replay import INITIAL, LABEL, LOOKBACK, simulate_hrp

from brain_api.core.patchtst.calendar import load_patchtst_calendar
from brain_api.core.sticky_selection import select_with_rank_band

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "patchtst_2025_snapshot_2026_replay"
REPO = ROOT.parents[1]
RUN_ID = "scratch:patchtst-2025-snapshot-2026-hrp-replay"


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def source_hashes(source):
    return {
        str(path.relative_to(source)): sha256(path)
        for path in sorted(source.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }


def load_evidence(source):
    summary = json.loads((source / "summary.json").read_text())
    manifest = json.loads((source / "price_manifest.json").read_text())
    if summary["snapshot"] != manifest["snapshot"]:
        raise ValueError("Original snapshot identities disagree")
    if summary["snapshot_latest_label"] >= "2026-01-01":
        raise ValueError("Snapshot contains 2026 training labels")
    prices = {}
    for name, digest in manifest["prices_sha256"].items():
        path = source / "data/prices" / name
        if sha256(path) != digest:
            raise ValueError(f"Frozen price evidence changed: {name}")
        frame = pd.read_csv(path, index_col="date", parse_dates=True)
        if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
            raise ValueError(f"Invalid price date index: {name}")
        prices[path.stem] = frame
    forecasts = pd.read_csv(source / "forecasts.csv", parse_dates=["monday", "anchor"])
    raw_ledger = json.loads((source / "weekly_ledger.json").read_text())
    plan, previous = [], None
    for week in raw_ledger:
        scores = forecasts.loc[forecasts.monday.eq(pd.Timestamp(week["monday"]))]
        if (
            not np.isfinite(scores.predicted_return).all()
            or scores.symbol.duplicated().any()
        ):
            raise ValueError("Original forecasts must be finite and unique")
        if not scores.anchor.eq(pd.Timestamp(week["anchor"])).all():
            raise ValueError("Forecast and ledger anchors disagree")
        selection = select_with_rank_band(
            scores.set_index("symbol").predicted_return.to_dict(), previous, 15, 20
        )
        if selection.selected != week["selected"] or len(selection.selected) != 15:
            raise ValueError("Saved selections disagree with original rank-band rule")
        plan.append(
            {
                k: pd.Timestamp(week[k])
                for k in ["monday", "anchor", "execution", "valuation"]
            }
            | {"selected": week["selected"], "week_complete": week["week_complete"]}
        )
        previous = set(selection.selected)
    original = pd.read_csv(
        source / "weekly_nav.csv", index_col="date", parse_dates=True
    )
    if len(original) != len(plan) + 1 or len(plan) != summary["weeks"]:
        raise ValueError("Original dates and weekly ledger disagree")
    return summary, manifest, prices, plan, original


def main():
    before = source_hashes(SOURCE)
    identity = {
        "source_sha256": before,
        "allocation": "HRP on exactly the same weekly PatchTST-selected 15",
        "daily_returns": LOOKBACK,
        "hrp_code_sha256": sha256(REPO / "brain_api/brain_api/core/hrp.py"),
        "runner_sha256": {
            name: sha256(ROOT / name) for name in ["run.py", "hrp_replay.py"]
        },
    }
    if (ROOT / "summary.json").exists():
        existing = json.loads((ROOT / "summary.json").read_text())
        if existing["identity"] != identity:
            raise ValueError(
                "Existing experiment differs; refusing to overwrite results"
            )
        for name, digest in existing["outputs_sha256"].items():
            if sha256(ROOT / name) != digest:
                raise ValueError(f"Completed result changed: {name}")
        print(
            "This exact HRP experiment is already complete; skipped without rerunning."
        )
        return
    original_summary, manifest, prices, plan, original = load_evidence(SOURCE)
    calendar = load_patchtst_calendar("us", date(2023, 11, 1), date(2026, 12, 10))
    nav, ledger, holdings, trades = simulate_hrp(prices, plan, calendar.sessions)
    if not nav.index.equals(original.index):
        raise ValueError("HRP marks differ from original comparison dates")
    combined = original.copy()
    combined[LABEL] = nav
    combined.to_csv(ROOT / "weekly_nav.csv")
    holdings.to_csv(ROOT / "hrp_weekly_holdings.csv", index=False)
    holdings.loc[holdings.monday.eq(plan[-1]["monday"])].to_csv(
        ROOT / "hrp_ending_holdings.csv", index=False
    )
    trades.to_csv(ROOT / "hrp_trades.csv", index=False)
    write_json(ROOT / "hrp_weekly_ledger.json", ledger)
    pd.DataFrame(
        [
            {
                "monday": w["monday"],
                "execution": w["execution"],
                "valuation": w["valuation"],
                "weekly_return": w["weekly_return"],
                "weekend_return": w["weekend_return"],
                "opening_nav": w["rebalance"]["nav_before"],
                "closing_nav": w["friday_close_nav"],
                "gross_turnover": w["rebalance"]["gross_turnover"],
                **w["history"],
            }
            for w in ledger
        ]
    ).to_csv(ROOT / "hrp_weekly_portfolio.csv", index=False)
    if source_hashes(SOURCE) != before:
        raise ValueError("Original experiment changed during the run")
    report = {
        "run_id": RUN_ID,
        "attempt": 1,
        "as_of": manifest["retrieved_at"],
        "run_at": datetime.now(UTC),
        "identity": identity,
        "source_experiment": str(SOURCE.relative_to(REPO)),
        "snapshot": original_summary["snapshot"],
        "weeks": len(plan),
        "completed_weeks": original_summary["completed_weeks"],
        "final_week_complete": original_summary["final_week_complete"],
        "first_execution": plan[0]["execution"],
        "last_valuation": plan[-1]["valuation"],
        "initial_capital": INITIAL,
        "transaction_cost": 0,
        "cash_buffer": 0,
        "top_n": 15,
        "hold_rank": 20,
        "lookback_daily_returns": LOOKBACK,
        "new_model_inference_runs": 0,
        "new_training_runs": 0,
        "metrics": {
            label: {
                "ending_value": float(combined[label].iloc[-1]),
                "total_return": float(combined[label].iloc[-1] / INITIAL - 1),
                "max_weekly_drawdown": float(
                    (combined[label] / combined[label].cummax() - 1).min()
                ),
            }
            for label in combined
        },
        "hrp_minus_equal_weight_return_pp": float(
            (nav.iloc[-1] - original.PatchTST.iloc[-1]) / INITIAL * 100
        ),
        "limitations": original_summary["limitations"]
        + [
            "Matched original frozen evidence ends November 25; final week is incomplete, not full-year 2026.",
            "HRP uses exactly 252 simple adjusted-close returns and unrounded production HRP weights; no current-week price, cash reserve or additional stock screen enters its target.",
        ],
        "outputs_sha256": {p.name: sha256(p) for p in ROOT.glob("*.csv")}
        | {"hrp_weekly_ledger.json": sha256(ROOT / "hrp_weekly_ledger.json")},
    }
    write_json(ROOT / "summary.json", report)
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()

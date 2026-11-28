"""Build transparent retrospective tables and keep recorded Monday targets separate."""

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SMOKE = ROOT.parent / "pi_manual_inference_2026_11_29"
observations = json.loads((ROOT / "observed_returns.json").read_text())["symbols"]
state = json.loads((ROOT / "pi_state.json").read_text())
alpha = json.loads((ROOT / "alpha_selection.json").read_text())
hrp = json.loads((ROOT / "alpha_hrp.json").read_text())
legacy = json.loads((SMOKE / "sac_inference_halal.json").read_text())["response"]
filtered = json.loads((ROOT / "filtered_sac_nov23.json").read_text())["response"]


def portfolio(weights, forecast=None):
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=5e-6):
        raise ValueError("Weights do not sum to 100%")
    rows = []
    for symbol, weight in weights.items():
        prices = (
            observations[symbol]
            if symbol != "CASH"
            else {
                "weekly_return_pct": 0.0,
                "monday_open_to_friday_close_pct": 0.0,
            }
        )
        row = {
            "symbol": symbol,
            "weight_pct": weight * 100,
            "stock_weekly_return_pct": prices["weekly_return_pct"],
            "stock_monday_open_to_friday_close_pct": prices[
                "monday_open_to_friday_close_pct"
            ],
            "weekly_contribution_pp": weight * prices["weekly_return_pct"],
            "monday_open_contribution_pp": weight
            * prices["monday_open_to_friday_close_pct"],
        }
        if forecast and symbol in forecast:
            row["patchtst_prediction_pct"] = forecast[symbol]
        rows.append(row)
    return {
        "rows": rows,
        "weekly_return_pct": sum(row["weekly_contribution_pp"] for row in rows),
        "monday_open_to_friday_close_pct": sum(
            row["monday_open_contribution_pp"] for row in rows
        ),
        "stock_winners": sum(
            row["stock_monday_open_to_friday_close_pct"] > 0
            for row in rows
            if row["symbol"] != "CASH"
        ),
        "stock_losers": sum(
            row["stock_monday_open_to_friday_close_pct"] < 0
            for row in rows
            if row["symbol"] != "CASH"
        ),
    }


forecasts = {row["symbol"]: row["forecast_pct"] for row in alpha["rows"]}
replay = {
    "alpha_hrp": portfolio(
        {s: w / 100 for s, w in hrp["percentage_weights"].items()}, forecasts
    ),
    "sac_halal": portfolio(legacy["target_weights"]),
    "sac_halal_filtered": portfolio(filtered["target_weights"]),
}
recorded = {}
recorded["alpha_hrp"] = portfolio(
    {
        row["stock"]: row["final_allocation_pct"] / 100
        for row in state["alpha_selected_history"]
        if row["year_week"] == "202648" and row["final_allocation_pct"] is not None
    }
)
for experience in state["experiences"]:
    data = experience["data"]
    key = "sac_" + data["universe"]
    recorded[key] = portfolio(data["action"])
    recorded[key]["model_version"] = data["model_version"]
    recorded[key]["has_execution_report"] = data["execution_report"] is not None
    recorded[key]["has_labeled_realized_return"] = data["realized_return"] is not None

result = {
    "decision_date": "2026-11-23",
    "observed_end": "2026-11-27",
    "forecast_anchor": alpha["forecast_anchor_date"],
    "forecast_end": alpha["forecast_end_date"],
    "model_versions": {
        "alpha_hrp": alpha["model_version"],
        "sac_halal": legacy["model_version"],
        "sac_halal_filtered": filtered["model_version"],
    },
    "method": {
        "replay": "Current models, hypothetical USD 10000 all-cash pre-decision portfolio; not models available on Monday.",
        "sticky": "Production prior week 202647 realized Stage-2 carry-set; pure K_in=15/K_hold=20 selector; no history writes.",
        "hrp": "Production core HRP, 252 returns, prices through November 20 only.",
        "weekly_return": "Adjusted November 27 close / adjusted November 20 close - 1.",
        "monday_entry": "Adjusted November 27 close / adjusted November 23 open - 1.",
        "portfolio_return": "Sum of target weight times each stock return; cash flat; fees, slippage and cash interest excluded.",
        "account_pnl": "Not calculated: target basket returns do not establish executed account performance.",
        "source": "Yahoo Finance through repository loader, auto_adjust=True; source price rows retained.",
    },
    "sticky": alpha["selection"],
    "replay": replay,
    "recorded_monday_target_baskets": recorded,
    "benchmark_spy": observations["SPY"],
}
(ROOT / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")

print("SUMMARY")
for name, item in replay.items():
    print(name, json.dumps({k: v for k, v in item.items() if k != "rows"}))
print("RECORDED MONDAY TARGET BASKETS")
for name, item in recorded.items():
    print(name, json.dumps({k: v for k, v in item.items() if k != "rows"}))
for name, item in replay.items():
    print("\n" + name)
    rows = item["rows"]
    if name == "alpha_hrp":
        order = {row["symbol"]: row["rank"] for row in alpha["rows"]}
        rows = sorted(rows, key=lambda row: order[row["symbol"]])
    else:
        rows = sorted(
            rows, key=lambda row: (row["symbol"] == "CASH", -row["weight_pct"])
        )
    for row in rows:
        prefix = (
            f"{row['patchtst_prediction_pct']:+.2f}% | "
            if "patchtst_prediction_pct" in row
            else ""
        )
        print(
            f"| {row['symbol']} | {prefix}{row['weight_pct']:.2f}% | {row['stock_monday_open_to_friday_close_pct']:+.2f}% | {row['stock_weekly_return_pct']:+.2f}% |"
        )

"""Fetch observed prices and calculate the production HRP sizing locally."""

import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

from brain_api.core.hrp import compute_hrp_allocation
from brain_api.core.prices import load_prices_yfinance

ROOT = Path(__file__).resolve().parent
SMOKE = ROOT.parent / "pi_manual_inference_2026_11_29"
selection = json.loads((ROOT / "alpha_selection.json").read_text())
state = json.loads((ROOT / "pi_state.json").read_text())
symbols = set(selection["selection"]["selected"])
for row in state["alpha_selected_history"]:
    symbols.add(row["stock"])
for experience in state["experiences"]:
    symbols.update(experience["data"]["action"])
for name in ("sac_inference_halal", "sac_inference_halal_filtered_2026-11-16"):
    data = json.loads((SMOKE / (name + ".json")).read_text())["response"]
    symbols.update(data["target_weights"])
symbols.discard("CASH")
symbols.add("SPY")
prices = load_prices_yfinance(
    sorted(symbols),
    date(2025, 8, 1),
    date(2026, 11, 27),
    required_columns=("open", "close"),
)
missing = sorted(symbols - set(prices))
if missing:
    raise RuntimeError("Missing price evidence: " + str(missing))
for symbol, frame in prices.items():
    frame.to_json(
        ROOT / ("prices_" + symbol + ".json"), orient="table", date_format="iso"
    )

selected_prices = {
    symbol: prices[symbol].loc[:"2026-11-20"]
    for symbol in selection["selection"]["selected"]
}
hrp = compute_hrp_allocation(
    selected_prices, lookback_days=252, min_data_days=60, as_of_date=date(2026, 11, 23)
)
if hrp.symbols_excluded:
    raise RuntimeError("HRP excluded selected names: " + str(hrp.symbols_excluded))
(ROOT / "alpha_hrp.json").write_text(json.dumps(asdict(hrp), indent=2) + "\n")
observations = {}
for symbol, frame in prices.items():
    normalized = frame.copy()
    if normalized.index.tz is not None:
        normalized.index = normalized.index.tz_localize(None)
    lookup = {str(index.date()): row for index, row in normalized.iterrows()}
    needed = ["2026-11-20", "2026-11-23", "2026-11-27"]
    absent = [day for day in needed if day not in lookup]
    if absent:
        observations[symbol] = {
            "missing_dates": absent,
            "last_available": str(normalized.index[-1].date()),
        }
        continue
    prior_close = float(lookup["2026-11-20"]["close"])
    monday_open = float(lookup["2026-11-23"]["open"])
    friday_close = float(lookup["2026-11-27"]["close"])
    observations[symbol] = {
        "adjusted_close_nov20": prior_close,
        "adjusted_open_nov23": monday_open,
        "adjusted_close_nov27": friday_close,
        "weekly_return_pct": 100 * (friday_close / prior_close - 1),
        "monday_open_to_friday_close_pct": 100 * (friday_close / monday_open - 1),
    }
(ROOT / "observed_returns.json").write_text(
    json.dumps(
        {
            "source": "Yahoo Finance via repository yfinance loader; auto_adjust=True",
            "start_close": "2026-11-20",
            "end_close": "2026-11-27",
            "symbols": observations,
        },
        indent=2,
    )
    + "\n"
)
print(
    json.dumps(
        {
            "hrp": asdict(hrp),
            "symbols_priced": len(observations),
            "missing_date_evidence": {
                s: r for s, r in observations.items() if "missing_dates" in r
            },
        },
        indent=2,
    )
)

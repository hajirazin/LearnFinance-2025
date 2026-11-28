"""Replay the current filtered SAC with scratch news and pre-open price evidence."""

import json
import math
import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SMOKE = ROOT.parent / "pi_manual_inference_2026_11_29"
spec = importlib.util.spec_from_file_location("pi_smoke_client", SMOKE / "client.py")
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)

client.ROOT = ROOT
model = json.loads((SMOKE / "active_halal_filtered.json").read_text())["response"]
forecast = json.loads((SMOKE / "full_patchtst_batch_us.json").read_text())["response"]
market = json.loads((SMOKE / "market_halal.json").read_text())["response"]
window = json.loads((SMOKE / "decision_window.json").read_text())["response"]
news = json.loads((ROOT / "filtered_news_nov23.json").read_text())
assert model["training_cutoff_date"] == "2024-11-15"
assert market["start_date"] == "2024-11-16" and market["as_of_date"] == "2026-11-23"
closes = {}
for symbol in model["symbols"]:
    frame = pd.read_json(ROOT / ("prices_" + symbol + ".json"), orient="table")
    if frame.index.tz is not None:
        frame.index = frame.index.tz_localize(None)
    closes[symbol] = [float(v) for v in frame.loc[:"2026-11-20", "close"].tail(253)]
    if len(closes[symbol]) != 253:
        raise ValueError("Missing pre-open price history: " + symbol)
body = {
    "portfolio": {"cash": 10000, "positions": []},
    "as_of": window["cutoff"],
    "as_of_date": "2026-11-23",
    "news_window": news,
    "feature_bundle": {
        "symbols": model["symbols"],
        "adjusted_closes": closes,
        "patchtst_forecasts": {
            s: forecast["scores"][s] / 100 for s in model["symbols"]
        },
        "market_history": market["rows"],
        "provenance": {
            "as_of_date": "2026-11-23",
            "adjusted_closes": {
                "provider": "yfinance",
                "price_basis": "adjusted",
                "last_completed_session": "2026-11-20",
                "lookback_bars": 253,
            },
            "market_history": market["provenance"],
            "patchtst": {
                k: forecast.get(k)
                for k in (
                    "as_of_date",
                    "model_version",
                    "forecast_recipe",
                    "forecast_anchor_date",
                    "forecast_end_date",
                )
            },
        },
    },
}
(ROOT / "filtered_sac_request_nov23.json").write_text(json.dumps(body, indent=2) + "\n")
result = client.call(
    "/inference/sac?universe=halal_filtered",
    payload=body,
    name="filtered_sac_nov23",
    timeout=600,
)
if result["model_version"] != model["model_version"]:
    raise ValueError("Active model changed during analysis")
weights = result["target_weights"]
if not all(math.isfinite(w) and w >= 0 for w in weights.values()) or not math.isclose(
    sum(weights.values()), 1
):
    raise ValueError("Invalid target weights")
print(
    json.dumps(
        {
            "model_version": result["model_version"],
            "weights": weights,
            "eligible": result["asset_eligibility"],
        },
        indent=2,
    ),
    flush=True,
)

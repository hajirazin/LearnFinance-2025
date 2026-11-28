"""Sequential inference-only Pi checks with complete real evidence."""

import json
import math
from datetime import date, timedelta

from client import ROOT, call

DAY = "2026-11-23"


def run():
    results = {}
    try:
        patch = call(
            "/inference/patchtst?universe=halal_new",
            payload={"as_of_date": DAY, "symbols": ["AAPL"]},
            name="patchtst_aapl_after_sync",
        )
        score = patch["predictions"][0]["predicted_weekly_return_pct"]
        if score is None or not math.isfinite(score):
            raise ValueError("AAPL forecast is missing or non-finite")
        results["patchtst"] = {"status": "passed", "response": patch}
    except Exception as exc:
        results["patchtst"] = {"status": "failed", "error": str(exc)}
    print(json.dumps({"patchtst": results["patchtst"]}), flush=True)

    window = call(
        "/calendar/monday-decision-window",
        payload={"run_date": DAY},
        name="decision_window_after_sync",
        timeout=30,
    )
    for universe in ("halal_filtered", "halal"):
        try:
            results[universe] = run_sac(universe, window)
        except Exception as exc:
            results[universe] = {"status": "failed", "error": str(exc)}
        print(json.dumps({universe: results[universe]}), flush=True)
        (ROOT / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    call("/health", name="health_after_checks", timeout=30)


def run_sac(universe, window, decision_date=DAY):
    tag = universe + "_" + decision_date
    model = call(
        "/models/active-symbols?universe=" + universe,
        name="active_after_sync_" + tag,
    )
    if decision_date <= model["training_cutoff_date"]:
        raise ValueError("Decision must be after the HMM training cutoff")
    symbols = model["symbols"]
    news = call(
        "/news/windows/query",
        payload={
            "symbols": symbols,
            "start_exclusive": window["start_exclusive"],
            "end_inclusive": window["end_inclusive"],
        },
        name="news_after_sync_" + tag,
    )
    forecast = call(
        "/inference/patchtst?universe=halal_new",
        payload={"as_of_date": decision_date, "symbols": symbols},
        name="forecast_" + tag,
    )
    # Historical smoke: request only prices available before Monday's pre-open
    # decision. This prevents the historical Monday close leaking into features.
    price_date = (date.fromisoformat(decision_date) - timedelta(days=1)).isoformat()
    prices = call(
        "/signals/prices",
        payload={"as_of_date": price_date, "symbols": symbols, "lookback_bars": 253},
        name="prices_" + tag,
    )
    start = (
        date.fromisoformat(model["training_cutoff_date"]) + timedelta(days=1)
    ).isoformat()
    market = call(
        "/signals/market-history",
        payload={"start_date": start, "as_of_date": decision_date},
        name="market_" + tag,
    )
    body = {
        "portfolio": {"cash": 10000, "positions": []},
        "as_of": window["cutoff"],
        "as_of_date": decision_date,
        "news_window": news,
        "feature_bundle": {
            "symbols": symbols,
            "adjusted_closes": prices["adjusted_closes"],
            "patchtst_forecasts": {
                row["symbol"]: row["predicted_weekly_return_pct"] / 100
                for row in forecast["predictions"]
                if row["predicted_weekly_return_pct"] is not None
            },
            "market_history": market["rows"],
            "provenance": {
                "as_of_date": decision_date,
                "adjusted_closes": {
                    **prices["provenance"],
                    "requested_as_of_date": price_date,
                },
                "market_history": market["provenance"],
                "patchtst": {
                    key: forecast.get(key)
                    for key in (
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
    (ROOT / ("sac_request_" + tag + ".json")).write_text(
        json.dumps(body, indent=2) + "\n"
    )
    result = call(
        "/inference/sac?universe=" + universe,
        payload=body,
        name="sac_inference_" + tag,
    )
    weights = result["target_weights"]
    if not all(math.isfinite(v) and v >= 0 for v in weights.values()):
        raise ValueError("Invalid allocation weights")
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-5):
        raise ValueError("Allocation weights do not sum to 1")
    return {
        "status": "passed",
        "decision_date": decision_date,
        "model_version": result["model_version"],
        "target_weights": weights,
        "asset_eligibility": result["asset_eligibility"],
        "state_digest": result["state_digest"],
    }


if __name__ == "__main__":
    run()

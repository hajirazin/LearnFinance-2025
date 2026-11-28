"""Check Monday's available inference inputs without trading or future news."""

import json
import math
from datetime import date, timedelta

from client import ROOT, call

DAY = "2026-11-30"
preflight = json.loads((ROOT / "full_patchtst_preflight.json").read_text())
summary = {"decision_date": DAY, "checks": {}}


def save():
    (ROOT / "next_monday_readiness.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


models = {}
for universe in ("halal_filtered", "halal"):
    models[universe] = call(
        "/models/active-symbols?universe=" + universe, name="monday_active_" + universe
    )
    summary["checks"]["model_" + universe] = {
        "version": models[universe]["model_version"],
        "symbols": len(models[universe]["symbols"]),
        "schema": models[universe]["sac_schema_version"],
    }
    save()

for market in ("us", "india"):
    universe = preflight["universes"][market]
    symbols = [stock["symbol"] for stock in universe["stocks"]]
    try:
        response = call(
            "/inference/patchtst/score-batch",
            payload={
                "market": market,
                "symbols": symbols,
                "as_of_date": DAY,
                "min_predictions": 15,
            },
            name="monday_patchtst_" + market,
            timeout=1200,
        )
        if len(response["scores"]) < 15 or not all(
            math.isfinite(v) for v in response["scores"].values()
        ):
            raise ValueError("Production score floor or finite-score validation failed")
        detail = {
            "status": "passed",
            "requested": len(symbols),
            "predicted": len(response["scores"]),
            "model_version": response["model_version"],
            "anchor": response.get("forecast_anchor_date"),
            "forecast_end": response.get("forecast_end_date"),
            "excluded": response["excluded_symbols"],
        }
        if market == "us":
            detail["sac_usable_forecasts"] = {
                u: len(set(m["symbols"]) & set(response["scores"]))
                for u, m in models.items()
            }
            for universe, count in detail["sac_usable_forecasts"].items():
                if count < 10:
                    raise ValueError("Fewer than 10 usable forecasts for " + universe)
        summary["checks"]["patchtst_" + market] = detail
    except Exception as exc:
        summary["checks"]["patchtst_" + market] = {
            "status": "failed",
            "error": str(exc),
        }
    print(json.dumps(summary["checks"]["patchtst_" + market]), flush=True)
    save()

symbols = sorted({symbol for model in models.values() for symbol in model["symbols"]})
prices = call(
    "/signals/prices",
    payload={
        "symbols": symbols,
        "as_of_date": DAY,
        "lookback_bars": 253,
    },
    name="monday_sac_prices",
    timeout=600,
)
summary["checks"]["sac_price_history"] = {
    u: {
        "usable_253_bars": sum(
            len(prices["adjusted_closes"].get(s, [])) >= 253 for s in m["symbols"]
        ),
        "required_assets": len(m["symbols"]),
    }
    for u, m in models.items()
}
save()
cutoffs = sorted({model["training_cutoff_date"] for model in models.values()})
for cutoff in cutoffs:
    start = (date.fromisoformat(cutoff) + timedelta(days=1)).isoformat()
    market = call(
        "/signals/market-history",
        payload={"start_date": start, "as_of_date": DAY},
        name="monday_market_" + cutoff,
        timeout=600,
    )
    summary["checks"]["market_" + cutoff] = {
        "status": "passed",
        "rows": len(market["rows"]),
        "last_session": market["rows"][-1]["date"] if market["rows"] else None,
    }
    save()
summary["checks"]["health"] = call(
    "/health", name="health_after_monday_readiness", timeout=30
)
summary["fresh_monday_news"] = (
    "Must be materialized after November 30 09:00 America/New_York; not fabricated or fetched early."
)
save()
print(json.dumps(summary, indent=2), flush=True)

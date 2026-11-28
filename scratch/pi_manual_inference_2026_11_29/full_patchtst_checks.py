"""Full-universe forecaster checks through the deployed production API."""

import json
import math

from client import ROOT, call

DAY = "2026-11-23"
preflight = json.loads((ROOT / "full_patchtst_preflight.json").read_text())
summary = {"decision_date": DAY, "checks": {}}
full = {}


def persist():
    (ROOT / "full_patchtst_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


def validate_symbols(symbols, market):
    if len(set(symbols)) != len(symbols):
        raise ValueError("Duplicate symbols in response")
    if market == "india" and any(not s.endswith(".NS") for s in symbols):
        raise ValueError("India symbol lost its .NS suffix")


def run_full(market, endpoint):
    response = call(
        endpoint,
        payload={"as_of_date": DAY},
        name="full_patchtst_" + market,
        timeout=1200,
    )
    predictions = response["predictions"]
    symbols = [p["symbol"] for p in predictions]
    validate_symbols(symbols, market)
    expected = preflight["models"][market].get("symbols", [])
    if response["model_version"] == preflight["models"][market].get("version"):
        if set(symbols) != set(expected):
            raise ValueError("Full-model response did not cover all metadata symbols")
    finite, excluded = {}, []
    for row in predictions:
        score = row["predicted_weekly_return_pct"]
        if score is None:
            excluded.append(row["symbol"])
        elif not math.isfinite(score):
            raise ValueError("Non-finite prediction")
        else:
            finite[row["symbol"]] = score
    if len(finite) < 15:
        raise ValueError("Fewer than 15 usable predictions")
    full[market] = finite
    return {
        "status": "passed",
        "endpoint": endpoint,
        "model_version": response["model_version"],
        "symbols_returned": len(symbols),
        "finite_predictions": len(finite),
        "excluded_symbols": excluded,
        "forecast_anchor_date": response.get("forecast_anchor_date"),
        "forecast_end_date": response.get("forecast_end_date"),
        "signals_used": response.get("signals_used"),
    }


def run_batch(market):
    universe = preflight["universes"][market]
    if not universe:
        raise ValueError("No existing production universe cache; no scrape started")
    symbols = [stock["symbol"] for stock in universe["stocks"]]
    validate_symbols(symbols, market)
    response = call(
        "/inference/patchtst/score-batch",
        payload={
            "market": market,
            "symbols": symbols,
            "as_of_date": DAY,
            "min_predictions": 15,
        },
        name="full_patchtst_batch_" + market,
        timeout=1200,
    )
    scores = response["scores"]
    excluded = response["excluded_symbols"]
    validate_symbols(list(scores), market)
    validate_symbols(excluded, market)
    if not all(math.isfinite(score) for score in scores.values()):
        raise ValueError("Non-finite batch score")
    if response["requested_count"] != len(symbols) or response[
        "predicted_count"
    ] != len(scores):
        raise ValueError("Batch response counts do not match request/results")
    if len(scores) < 15 or set(scores) & set(excluded):
        raise ValueError("Batch prediction floor or exclusion invariant failed")
    if set(scores) | set(excluded) != set(symbols):
        raise ValueError("Batch scores/exclusions do not account for all symbols")
    comparison = {}
    if market in full:
        common = set(full[market]) & set(scores)
        diffs = [abs(full[market][s] - scores[s]) for s in common]
        comparison = {
            "common_symbols": len(common),
            "max_abs_score_difference_pct": max(diffs, default=0.0),
        }
    return {
        "status": "passed",
        "endpoint": "/inference/patchtst/score-batch",
        "market": market,
        "model_version": response["model_version"],
        "requested_count": len(symbols),
        "finite_predictions": len(scores),
        "excluded_symbols": excluded,
        "forecast_anchor_date": response.get("forecast_anchor_date"),
        "forecast_end_date": response.get("forecast_end_date"),
        "full_route_comparison": comparison,
    }


for name, operation in (
    ("us_full", lambda: run_full("us", "/inference/patchtst")),
    ("india_full", lambda: run_full("india", "/inference/patchtst/india")),
    ("us_batch", lambda: run_batch("us")),
    ("india_batch", lambda: run_batch("india")),
):
    try:
        result = operation()
    except Exception as exc:
        result = {"status": "failed", "error": str(exc)}
    summary["checks"][name] = result
    persist()
    print(json.dumps({name: result}), flush=True)

summary["health_after"] = call("/health", name="health_after_full_patchtst", timeout=30)
persist()

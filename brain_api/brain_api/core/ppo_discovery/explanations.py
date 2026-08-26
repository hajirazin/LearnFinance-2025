"""Deterministic explanations for ppo_discovery. The LLM cannot change weights."""

from __future__ import annotations

from typing import Any

from brain_api.core.ppo_discovery.config import (
    ASSET_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
)
from brain_api.core.ppo_discovery.schemas import CanonicalPPOState


def build_explanations(
    state: CanonicalPPOState,
    weights: dict[str, float],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Facts only: model inputs plus audit-only news stats labeled as such."""
    selected = [
        symbol for symbol, weight in weights.items() if symbol != "CASH" and weight > 0
    ]
    per_asset = []
    for symbol in selected:
        index = state.symbols.index(symbol)
        features = {
            name: float(state.asset_features[index, feature_i])
            for feature_i, name in enumerate(ASSET_FEATURE_NAMES)
        }
        audit = (state.audit_news.get("per_symbol") or {}).get(symbol, {})
        per_asset.append(
            {
                "symbol": symbol,
                "weight": float(weights[symbol]),
                "model_inputs": features,
                "audit_only": {
                    "average_confidence": audit.get("average_confidence"),
                    "sentiment_dispersion": audit.get("sentiment_dispersion"),
                    "unique_source_count": audit.get("unique_source_count"),
                    "label": "audit data, not a model input",
                },
            }
        )
    globals_ = {
        name: float(state.globals[index])
        for index, name in enumerate(GLOBAL_FEATURE_NAMES)
    }
    return {
        "model_version": metadata.get("version"),
        "k": len(selected),
        "cash_weight": float(weights.get("CASH", 0.0)),
        "globals": globals_,
        "per_asset": per_asset,
        "exclusions": state.exclusions,
        "held_symbols": list(state.held_symbols),
        "survivorship_bias": (
            "Historical training applies today's halal_new roster retrospectively."
        ),
        "news_contract": "News evidence is mandatory; incomplete queries abort the week.",
    }


__all__ = ["build_explanations"]

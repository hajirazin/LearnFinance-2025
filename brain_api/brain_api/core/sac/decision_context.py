"""Canonical SAC decision inputs and auditable state snapshots."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np

SAC_SIGNAL_NAMES = (
    "news_sentiment",
    "news_coverage",
    "gross_margin",
    "debt_to_equity",
    "fundamental_age",
    "momentum_1w",
    "momentum_4w",
    "momentum_12_1",
    "earnings_yield",
)

# Backward-compatible alias for callers/tests still importing the old name.
SAC_V2_SIGNAL_NAMES = SAC_SIGNAL_NAMES


class SACDecisionContextError(ValueError):
    """Raised when a canonical SAC decision input is incomplete or invalid."""


def _finite_float(value: Any, *, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SACDecisionContextError(f"{field} must be a number") from exc
    if not math.isfinite(parsed):
        raise SACDecisionContextError(f"{field} must be finite")
    return parsed


def _require_exact_symbols(
    values: Mapping[str, Any],
    symbols: tuple[str, ...],
    *,
    field: str,
) -> None:
    expected = set(symbols)
    actual = set(values)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise SACDecisionContextError(
            f"{field} symbol mismatch: missing={missing}, extra={extra}"
        )


@dataclass(frozen=True)
class SACFeatureBundle:
    """The exact point-in-time signals and forecasts supplied to SAC."""

    symbols: tuple[str, ...]
    signals: dict[str, dict[str, float]]
    patchtst_forecasts: dict[str, float]
    provenance: dict[str, Any]

    @classmethod
    def create(
        cls,
        *,
        symbols: list[str] | tuple[str, ...],
        signals: Mapping[str, Mapping[str, Any]],
        patchtst_forecasts: Mapping[str, Any],
        provenance: Mapping[str, Any] | None = None,
    ) -> SACFeatureBundle:
        """Validate and normalize the live PatchTST-only feature bundle."""
        symbol_order = tuple(symbols)
        if not symbol_order or len(set(symbol_order)) != len(symbol_order):
            raise SACDecisionContextError("symbols must be non-empty and unique")
        _require_exact_symbols(signals, symbol_order, field="signals")
        _require_exact_symbols(
            patchtst_forecasts, symbol_order, field="patchtst_forecasts"
        )

        normalized_signals: dict[str, dict[str, float]] = {}
        required = set(SAC_SIGNAL_NAMES)
        for symbol in symbol_order:
            symbol_signals = signals[symbol]
            missing = sorted(required - set(symbol_signals))
            if missing:
                raise SACDecisionContextError(
                    f"signals[{symbol}] missing required features: {missing}"
                )
            normalized_signals[symbol] = {
                name: _finite_float(
                    symbol_signals[name], field=f"signals[{symbol}].{name}"
                )
                for name in SAC_SIGNAL_NAMES
            }
            coverage = normalized_signals[symbol]["news_coverage"]
            if not 0.0 <= coverage <= 1.0:
                raise SACDecisionContextError(
                    f"signals[{symbol}].news_coverage must be between 0 and 1"
                )

        return cls(
            symbols=symbol_order,
            signals=normalized_signals,
            patchtst_forecasts={
                symbol: _finite_float(
                    patchtst_forecasts[symbol],
                    field=f"patchtst_forecasts[{symbol}]",
                )
                for symbol in symbol_order
            },
            provenance=dict(provenance or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation with stable symbol order."""
        return {
            "symbols": list(self.symbols),
            "signals": self.signals,
            "patchtst_forecasts": self.patchtst_forecasts,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class SACDecisionContext:
    """Canonical decision context owned by Brain and supplied by Temporal."""

    as_of_date: date
    feature_bundle: SACFeatureBundle
    current_weights: dict[str, float]

    @classmethod
    def create(
        cls,
        *,
        as_of_date: date,
        feature_bundle: SACFeatureBundle,
        current_weights: Mapping[str, Any],
    ) -> SACDecisionContext:
        """Validate portfolio weights against the feature-bundle symbol order."""
        expected = (*feature_bundle.symbols, "CASH")
        _require_exact_symbols(current_weights, expected, field="current_weights")
        normalized = {
            symbol: _finite_float(
                current_weights[symbol], field=f"current_weights[{symbol}]"
            )
            for symbol in expected
        }
        negative = [symbol for symbol, value in normalized.items() if value < 0]
        if negative:
            raise SACDecisionContextError(
                f"current_weights must be nonnegative: {negative}"
            )
        total = sum(normalized.values())
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise SACDecisionContextError(
                f"current_weights must sum to 1.0, got {total:.12f}"
            )
        return cls(
            as_of_date=as_of_date,
            feature_bundle=feature_bundle,
            current_weights=normalized,
        )

    def weight_array(self) -> np.ndarray:
        """Return weights in model symbol order with cash last."""
        return np.asarray(
            [
                *(
                    self.current_weights[symbol]
                    for symbol in self.feature_bundle.symbols
                ),
                self.current_weights["CASH"],
            ],
            dtype=np.float64,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation used for auditing."""
        return {
            "as_of_date": self.as_of_date.isoformat(),
            "feature_bundle": self.feature_bundle.to_dict(),
            "current_weights": self.current_weights,
        }


@dataclass(frozen=True)
class SACDecisionState:
    """The exact actor state vector and its deterministic audit digest."""

    vector: tuple[float, ...]
    context: SACDecisionContext
    digest: str

    @classmethod
    def create(
        cls,
        *,
        vector: np.ndarray,
        context: SACDecisionContext,
    ) -> SACDecisionState:
        """Create a finite state snapshot and SHA-256 digest."""
        flat = np.asarray(vector, dtype=np.float64)
        if flat.ndim != 1 or not np.all(np.isfinite(flat)):
            raise SACDecisionContextError(
                "SAC decision state vector must be 1-D finite"
            )
        snapshot = {
            "vector": [float(value) for value in flat],
            "context": context.to_dict(),
        }
        canonical = json.dumps(
            snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return cls(
            vector=tuple(snapshot["vector"]),
            context=context,
            digest=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the snapshot shape persisted with SAC experience."""
        return {
            "vector": list(self.vector),
            "context": self.context.to_dict(),
            "digest": self.digest,
        }

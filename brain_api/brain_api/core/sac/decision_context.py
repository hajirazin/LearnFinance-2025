"""Canonical raw evidence and auditable state snapshots for SAC v3."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np

from brain_api.core.portfolio_rl.state import MAX_ASSETS, MIN_ELIGIBLE_ASSETS
from brain_api.core.sac.momentum_signals import (
    MomentumSignalError,
    compute_momentum_1w,
    compute_momentum_4w,
    compute_momentum_12_1,
    compute_realized_vol_20d,
)

SAC_SIGNAL_NAMES = (
    "news_sentiment",
    "momentum_1w",
    "momentum_4w",
    "momentum_12_1",
    "realized_vol_20d",
)


class SACDecisionContextError(ValueError):
    """Raised when a SAC v3 decision input is incomplete or invalid."""


def _finite_float(value: Any, *, field: str, positive: bool = False) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SACDecisionContextError(f"{field} must be a number") from exc
    if not math.isfinite(parsed) or (positive and parsed <= 0):
        qualifier = "finite and positive" if positive else "finite"
        raise SACDecisionContextError(f"{field} must be {qualifier}")
    return parsed


def _require_exact_symbols(
    values: Mapping[str, Any], symbols: tuple[str, ...], *, field: str
) -> None:
    expected, actual = set(symbols), set(values)
    if expected != actual:
        raise SACDecisionContextError(
            f"{field} symbol mismatch: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


@dataclass(frozen=True)
class SACFeatureBundle:
    """Point-in-time raw evidence supplied by Temporal; no ranking/math lives there."""

    symbols: tuple[str, ...]
    adjusted_closes: dict[str, tuple[float, ...]]
    news_sentiment: dict[str, float]
    news_article_counts: dict[str, int]
    patchtst_forecasts: dict[str, float | None]
    market_history: tuple[dict[str, Any], ...]
    provenance: dict[str, Any]

    @classmethod
    def create(
        cls,
        *,
        symbols: list[str] | tuple[str, ...],
        adjusted_closes: Mapping[str, list[float]],
        news_sentiment: Mapping[str, Any],
        news_article_counts: Mapping[str, Any],
        patchtst_forecasts: Mapping[str, Any],
        market_history: list[dict[str, Any]],
        provenance: Mapping[str, Any] | None = None,
    ) -> SACFeatureBundle:
        symbol_order = tuple(symbols)
        if not 1 <= len(symbol_order) <= MAX_ASSETS or len(set(symbol_order)) != len(
            symbol_order
        ):
            raise SACDecisionContextError(
                f"symbols must contain 1..{MAX_ASSETS} unique values"
            )
        # News is provider-checked and therefore must be exact; a missing row is
        # a provider/gap failure, never eligibility or neutral sentiment.
        _require_exact_symbols(news_sentiment, symbol_order, field="news_sentiment")
        _require_exact_symbols(
            news_article_counts, symbol_order, field="news_article_counts"
        )
        normalized_sentiment: dict[str, float] = {}
        normalized_counts: dict[str, int] = {}
        for symbol in symbol_order:
            try:
                count = int(news_article_counts[symbol])
            except (TypeError, ValueError) as exc:
                raise SACDecisionContextError(
                    f"news_article_counts[{symbol}] must be an integer"
                ) from exc
            if count < 0:
                raise SACDecisionContextError(
                    f"news_article_counts[{symbol}] must be nonnegative"
                )
            sentiment = _finite_float(
                news_sentiment[symbol], field=f"news_sentiment[{symbol}]"
            )
            if count == 0 and sentiment != 0.0:
                raise SACDecisionContextError(
                    f"news_sentiment[{symbol}] must be zero for a genuine zero-article observation"
                )
            normalized_counts[symbol] = count
            normalized_sentiment[symbol] = sentiment

        closes: dict[str, tuple[float, ...]] = {}
        for symbol, values in adjusted_closes.items():
            if symbol not in symbol_order:
                raise SACDecisionContextError(
                    f"adjusted_closes contains extra symbol {symbol}"
                )
            try:
                closes[symbol] = tuple(float(value) for value in values)
            except (TypeError, ValueError) as exc:
                raise SACDecisionContextError(
                    f"adjusted_closes[{symbol}] must be numeric"
                ) from exc

        forecasts: dict[str, float | None] = {}
        for symbol in symbol_order:
            forecast = patchtst_forecasts.get(symbol)
            forecasts[symbol] = (
                float(forecast)
                if forecast is not None and math.isfinite(float(forecast))
                else None
            )
        return cls(
            symbols=symbol_order,
            adjusted_closes=closes,
            news_sentiment=normalized_sentiment,
            news_article_counts=normalized_counts,
            patchtst_forecasts=forecasts,
            market_history=tuple(dict(row) for row in market_history),
            provenance=dict(provenance or {}),
        )

    def eligible_inputs(
        self, current_weights: Mapping[str, float], *, production: bool = True
    ) -> tuple[np.ndarray, dict[str, dict[str, float]], dict[str, float]]:
        """Compute raw features and eligibility; never impute missing evidence."""
        mask = np.zeros(MAX_ASSETS, dtype=bool)
        signals: dict[str, dict[str, float]] = {}
        forecasts: dict[str, float] = {}
        for index, symbol in enumerate(self.symbols):
            forecast = self.patchtst_forecasts[symbol]
            closes = self.adjusted_closes.get(symbol)
            if forecast is None or closes is None:
                continue
            try:
                index_as_of = len(closes) - 1
                raw_signals = {
                    "news_sentiment": self.news_sentiment[symbol],
                    "momentum_1w": compute_momentum_1w(closes, as_of_index=index_as_of),
                    "momentum_4w": compute_momentum_4w(closes, as_of_index=index_as_of),
                    "momentum_12_1": compute_momentum_12_1(
                        closes, as_of_index=index_as_of
                    ),
                    "realized_vol_20d": compute_realized_vol_20d(
                        closes, as_of_index=index_as_of
                    ),
                }
            except MomentumSignalError:
                continue
            mask[index] = True
            signals[symbol] = raw_signals
            forecasts[symbol] = forecast
        n_valid = int(mask.sum())
        if production and n_valid < MIN_ELIGIBLE_ASSETS:
            raise SACDecisionContextError(
                f"SAC v3 requires at least {MIN_ELIGIBLE_ASSETS} eligible assets; got {n_valid}"
            )
        if n_valid == 0:
            raise SACDecisionContextError("SAC v3 has no eligible assets")
        return mask, signals, forecasts

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbols": list(self.symbols),
            "adjusted_closes": {
                key: list(value) for key, value in self.adjusted_closes.items()
            },
            "news_sentiment": self.news_sentiment,
            "news_article_counts": self.news_article_counts,
            "patchtst_forecasts": self.patchtst_forecasts,
            "market_history": list(self.market_history),
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class SACDecisionContext:
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
        expected = (*feature_bundle.symbols, "CASH")
        _require_exact_symbols(current_weights, expected, field="current_weights")
        normalized = {
            symbol: _finite_float(
                current_weights[symbol], field=f"current_weights[{symbol}]"
            )
            for symbol in expected
        }
        if any(value < 0 for value in normalized.values()) or not math.isclose(
            sum(normalized.values()), 1.0, rel_tol=0.0, abs_tol=1e-6
        ):
            raise SACDecisionContextError(
                "current_weights must be a nonnegative simplex"
            )
        return cls(as_of_date, feature_bundle, normalized)

    def weight_array(self) -> np.ndarray:
        weights = np.zeros(MAX_ASSETS + 1, dtype=np.float64)
        for index, symbol in enumerate(self.feature_bundle.symbols):
            weights[index] = self.current_weights[symbol]
        weights[-1] = self.current_weights["CASH"]
        return weights

    def to_dict(self) -> dict[str, Any]:
        return {
            "as_of_date": self.as_of_date.isoformat(),
            "feature_bundle": self.feature_bundle.to_dict(),
            "current_weights": self.current_weights,
        }


@dataclass(frozen=True)
class SACDecisionState:
    vector: tuple[float, ...]
    context: SACDecisionContext
    digest: str

    @classmethod
    def create(
        cls, *, vector: np.ndarray, context: SACDecisionContext
    ) -> SACDecisionState:
        flat = np.asarray(vector, dtype=np.float64)
        if flat.ndim != 1 or not np.all(np.isfinite(flat)):
            raise SACDecisionContextError(
                "SAC decision state vector must be 1-D finite"
            )
        snapshot = {"vector": list(map(float, flat)), "context": context.to_dict()}
        canonical = json.dumps(
            snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return cls(
            tuple(snapshot["vector"]),
            context,
            hashlib.sha256(canonical.encode()).hexdigest(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "vector": list(self.vector),
            "context": self.context.to_dict(),
            "digest": self.digest,
        }

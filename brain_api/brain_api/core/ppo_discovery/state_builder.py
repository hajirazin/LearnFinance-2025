"""Build the compact 9+7 canonical ppo_discovery state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from brain_api.core.ppo_discovery.config import (
    ASSET_FEATURE_NAMES,
    AUDIT_NEWS_FIELDS,
    ENCODER_CHANNELS,
    ENCODER_SESSIONS,
    EXPLICIT_ASSET_FEATURES,
    GLOBAL_FEATURE_NAMES,
    GLOBAL_FEATURES,
    MAX_ASSETS,
    MIN_ELIGIBLE_ASSETS,
)
from brain_api.core.ppo_discovery.price_features import (
    apply_encoder_channel_scaler,
    encoder_channels_from_ohlcv,
    explicit_price_signals,
    rank_eligible,
    spy_return_20d,
    validate_ohlcv_frame,
)
from brain_api.core.ppo_discovery.schemas import (
    CanonicalPPOState,
    PPODiscoveryError,
    SymbolNewsFeatures,
    UniverseSnapshot,
    assert_feature_contract,
    sha256_digest,
    state_to_digest_payload,
)

assert_feature_contract()


@dataclass(frozen=True)
class StateBuildRequest:
    """Inputs required to pack one decision state."""

    as_of: datetime
    universe_snapshot: UniverseSnapshot
    ohlcv_by_symbol: Mapping[str, pd.DataFrame]
    news_by_symbol: Mapping[str, SymbolNewsFeatures]
    current_weights: Mapping[str, float]
    p_calm: float
    p_stress: float
    spy_closes: Sequence[float]
    feature_scalers: Mapping[str, Any] | None = None
    tradable_symbols: frozenset[str] | None = None


def _cash_weight(weights: Mapping[str, float]) -> float:
    cash = float(weights.get("CASH", 0.0))
    if not np.isfinite(cash) or cash < 0:
        raise PPODiscoveryError("CASH weight must be finite and nonnegative")
    return cash


def build_ppo_discovery_state(request: StateBuildRequest) -> CanonicalPPOState:
    """Pack padded tensors. Incomplete news or held-price gaps abort."""
    symbols = request.universe_snapshot.sorted_symbols
    if len(symbols) > MAX_ASSETS:
        raise PPODiscoveryError("universe exceeds 512-slot capacity")
    held = tuple(
        sorted(
            symbol
            for symbol, weight in request.current_weights.items()
            if symbol != "CASH" and float(weight) > 0
        )
    )
    exclusions: dict[str, str] = {}
    eligible: list[str] = []
    histories: dict[str, np.ndarray] = {}
    price_signals: dict[str, dict[str, float]] = {}

    for symbol in symbols:
        news = request.news_by_symbol.get(symbol)
        if news is None or not news.query_complete:
            raise PPODiscoveryError(f"news query incomplete for {symbol}")
        if (
            request.tradable_symbols is not None
            and symbol not in request.tradable_symbols
        ):
            exclusions[symbol] = "not_alpaca_tradable"
            continue
        frame = request.ohlcv_by_symbol.get(symbol)
        try:
            if frame is None:
                raise PPODiscoveryError(f"{symbol} missing OHLCV")
            validated = validate_ohlcv_frame(symbol, frame)
            history = apply_encoder_channel_scaler(
                encoder_channels_from_ohlcv(validated), request.feature_scalers
            )
            closes = validated["close"].to_numpy(dtype=np.float64)
            signals = explicit_price_signals(closes)
        except PPODiscoveryError as exc:
            if symbol in held:
                raise PPODiscoveryError(
                    f"held asset {symbol} lacks a finite positive execution price: {exc}"
                ) from exc
            exclusions[symbol] = str(exc)
            continue
        eligible.append(symbol)
        histories[symbol] = history
        price_signals[symbol] = signals

    if len(eligible) < MIN_ELIGIBLE_ASSETS:
        raise PPODiscoveryError(
            f"need at least {MIN_ELIGIBLE_ASSETS} eligible assets, got {len(eligible)}"
        )

    mom1 = rank_eligible(
        {symbol: price_signals[symbol]["momentum_1w"] for symbol in eligible}, eligible
    )
    mom4 = rank_eligible(
        {symbol: price_signals[symbol]["momentum_4w"] for symbol in eligible}, eligible
    )
    mom12 = rank_eligible(
        {symbol: price_signals[symbol]["momentum_12_1"] for symbol in eligible},
        eligible,
    )
    vol = rank_eligible(
        {symbol: price_signals[symbol]["realized_vol_20d"] for symbol in eligible},
        eligible,
    )
    news_raw = {
        symbol: request.news_by_symbol[symbol].raw_sentiment for symbol in eligible
    }
    news_rank = rank_eligible(news_raw, eligible)

    log_counts = np.asarray(
        [request.news_by_symbol[symbol].log1p_article_count for symbol in eligible],
        dtype=np.float64,
    )
    if request.feature_scalers and "log1p_article_count" in request.feature_scalers:
        mean = float(request.feature_scalers["log1p_article_count"]["mean"])
        scale = float(request.feature_scalers["log1p_article_count"]["scale"])
        if scale <= 0 or not np.isfinite(scale) or not np.isfinite(mean):
            raise PPODiscoveryError("invalid log1p_article_count scaler")
        log_counts = (log_counts - mean) / scale
    scaled_counts = {
        symbol: float(log_counts[index]) for index, symbol in enumerate(eligible)
    }

    price_history = np.zeros(
        (MAX_ASSETS, ENCODER_SESSIONS, ENCODER_CHANNELS), dtype=np.float64
    )
    asset_features = np.zeros((MAX_ASSETS, EXPLICIT_ASSET_FEATURES), dtype=np.float64)
    asset_mask = np.zeros(MAX_ASSETS, dtype=bool)
    padded_symbols = list(symbols) + [""] * (MAX_ASSETS - len(symbols))
    eligible_set = set(eligible)
    for index, symbol in enumerate(symbols):
        if symbol not in eligible_set:
            continue
        asset_mask[index] = True
        price_history[index] = histories[symbol]
        weight = float(request.current_weights.get(symbol, 0.0))
        news = request.news_by_symbol[symbol]
        asset_features[index] = np.asarray(
            [
                mom1[symbol],
                mom4[symbol],
                mom12[symbol],
                vol[symbol],
                news_rank[symbol],
                news.raw_sentiment,
                scaled_counts[symbol],
                news.news_recency,
                weight,
            ],
            dtype=np.float64,
        )

    raw_news_values = np.asarray(
        [news_raw[symbol] for symbol in eligible], dtype=np.float64
    )
    has_news = np.asarray(
        [request.news_by_symbol[symbol].has_news for symbol in eligible],
        dtype=np.float64,
    )
    mom4_raw = np.asarray(
        [price_signals[symbol]["momentum_4w"] for symbol in eligible], dtype=np.float64
    )
    globals_ = np.asarray(
        [
            float(request.p_calm),
            float(request.p_stress),
            _cash_weight(request.current_weights),
            float(np.mean(mom4_raw > 0)),
            spy_return_20d(request.spy_closes),
            float(np.median(raw_news_values)),
            float(np.mean(has_news)),
        ],
        dtype=np.float64,
    )
    if globals_.shape != (GLOBAL_FEATURES,):
        raise PPODiscoveryError("global feature width drifted")
    if asset_features.shape[-1] != len(ASSET_FEATURE_NAMES):
        raise PPODiscoveryError("asset feature width drifted")
    if any(name in ASSET_FEATURE_NAMES for name in AUDIT_NEWS_FIELDS):
        raise PPODiscoveryError("audit news leaked into asset features")

    negative_frac = float(np.mean(raw_news_values < 0))
    positive_frac = float(np.mean(raw_news_values > 0))
    news_dispersion = (
        float(np.std(raw_news_values, ddof=0)) if len(raw_news_values) >= 2 else 0.0
    )
    audit_news = {
        "fraction_negative_news": negative_frac,
        "fraction_positive_news": positive_frac,
        "news_sentiment_dispersion": news_dispersion,
        "per_symbol": {
            symbol: {
                "average_confidence": request.news_by_symbol[symbol].average_confidence,
                "sentiment_dispersion": request.news_by_symbol[
                    symbol
                ].sentiment_dispersion,
                "unique_source_count": request.news_by_symbol[
                    symbol
                ].unique_source_count,
                "has_news": request.news_by_symbol[symbol].has_news,
            }
            for symbol in eligible
        },
    }
    evidence_manifest = {
        "eligible": eligible,
        "excluded": exclusions,
        "universe_snapshot_sha256": request.universe_snapshot.snapshot_sha256,
        "news_complete": True,
        "asset_feature_names": list(ASSET_FEATURE_NAMES),
        "global_feature_names": list(GLOBAL_FEATURE_NAMES),
    }
    state = CanonicalPPOState(
        symbols=tuple(padded_symbols),
        asset_mask=asset_mask,
        price_history=price_history,
        asset_features=asset_features,
        globals=globals_,
        current_weights=dict(request.current_weights),
        news_by_symbol=dict(request.news_by_symbol),
        audit_news=audit_news,
        exclusions=exclusions,
        universe_snapshot=request.universe_snapshot,
        evidence_manifest=evidence_manifest,
        state_digest="",
        as_of=request.as_of.isoformat(),
        held_symbols=held,
    )
    digest = sha256_digest(state_to_digest_payload(state))
    state.state_digest = digest
    evidence_manifest["state_digest"] = digest
    state.evidence_manifest = evidence_manifest
    return state


__all__ = ["StateBuildRequest", "build_ppo_discovery_state"]

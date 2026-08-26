"""Tiny synthetic states for CI and endpoint smoke tests. Not a live fallback."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from brain_api.core.ppo_discovery.config import HISTORY_BARS, MIN_ELIGIBLE_ASSETS
from brain_api.core.ppo_discovery.schemas import CanonicalPPOState, SymbolNewsFeatures
from brain_api.core.ppo_discovery.state_builder import (
    StateBuildRequest,
    build_ppo_discovery_state,
)
from brain_api.core.ppo_discovery.universe_snapshot import build_universe_snapshot


def synthetic_ohlcv(
    n: int = HISTORY_BARS, seed: int = 0, start: float = 100.0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.01, size=n)
    close = start * np.cumprod(1.0 + rets)
    open_ = np.concatenate([[start], close[:-1]])
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.0, 0.01, size=n))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.0, 0.01, size=n))
    volume = rng.uniform(1e5, 2e5, size=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


def synthetic_news(
    symbol: str, *, raw: float = 0.1, count: int = 2
) -> SymbolNewsFeatures:
    return SymbolNewsFeatures(
        symbol=symbol,
        raw_sentiment=raw,
        article_count=count,
        average_confidence=0.8,
        sentiment_dispersion=0.05,
        hours_since_latest=12.0 if count else 0.0,
        unique_source_count=1 if count else 0,
        has_news=1 if count else 0,
        query_complete=True,
        news_recency=float(np.exp(-12.0 / 168.0)) if count else 0.0,
        log1p_article_count=float(np.log1p(count)),
        article_ids_sha256="sha256:abc",
        request_manifest_sha256="sha256:def",
    )


def make_synthetic_state(
    n_assets: int = MIN_ELIGIBLE_ASSETS,
    *,
    current_weights: dict[str, float] | None = None,
) -> CanonicalPPOState:
    symbols = [f"S{i:02d}" for i in range(n_assets)]
    snapshot = build_universe_snapshot(
        symbols, retrieved_at=datetime(2026, 8, 31, tzinfo=UTC)
    )
    news = {
        symbol: synthetic_news(symbol, raw=0.1 + i * 0.01, count=i + 1)
        for i, symbol in enumerate(snapshot.sorted_symbols)
    }
    ohlcv = {
        symbol: synthetic_ohlcv(seed=i + 1)
        for i, symbol in enumerate(snapshot.sorted_symbols)
    }
    spy = synthetic_ohlcv(seed=99)["close"].to_numpy()[-30:]
    return build_ppo_discovery_state(
        StateBuildRequest(
            as_of=datetime(2026, 8, 31, 13, 0, tzinfo=UTC),
            universe_snapshot=snapshot,
            ohlcv_by_symbol=ohlcv,
            news_by_symbol=news,
            current_weights=current_weights or {"CASH": 1.0},
            p_calm=0.4,
            p_stress=0.3,
            spy_closes=spy,
        )
    )


__all__ = ["make_synthetic_state", "synthetic_news", "synthetic_ohlcv"]

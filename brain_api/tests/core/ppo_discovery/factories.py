"""Shared synthetic fixtures for ppo_discovery tests."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from brain_api.core.ppo_discovery.config import HISTORY_BARS, MIN_ELIGIBLE_ASSETS
from brain_api.core.ppo_discovery.schemas import SymbolNewsFeatures, UniverseSnapshot
from brain_api.core.ppo_discovery.universe_snapshot import build_universe_snapshot


def make_ohlcv(
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


def make_news(symbol: str, *, raw: float = 0.1, count: int = 2) -> SymbolNewsFeatures:
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


def make_snapshot(n: int = MIN_ELIGIBLE_ASSETS) -> UniverseSnapshot:
    symbols = [f"S{i:02d}" for i in range(n)]
    return build_universe_snapshot(
        symbols, retrieved_at=datetime(2026, 8, 31, tzinfo=UTC)
    )


def verified_zero_news(symbol: str) -> SymbolNewsFeatures:
    return make_news(symbol, raw=0.0, count=0)

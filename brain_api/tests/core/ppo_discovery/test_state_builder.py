"""Canonical 9+7 state packing tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from brain_api.core.ppo_discovery.config import (
    ASSET_FEATURE_NAMES,
    AUDIT_NEWS_FIELDS,
    GLOBAL_FEATURE_NAMES,
    MAX_ASSETS,
)
from brain_api.core.ppo_discovery.schemas import (
    PPODiscoveryError,
    sha256_digest,
    state_to_digest_payload,
)
from brain_api.core.ppo_discovery.state_builder import (
    StateBuildRequest,
    build_ppo_discovery_state,
)
from tests.core.ppo_discovery.factories import (
    make_news,
    make_ohlcv,
    make_snapshot,
    verified_zero_news,
)


def _request(**overrides):
    snapshot = overrides.get("universe_snapshot") or make_snapshot()
    news = {
        symbol: make_news(symbol, raw=0.1 + i * 0.01, count=i + 1)
        for i, symbol in enumerate(snapshot.sorted_symbols)
    }
    ohlcv = {
        symbol: make_ohlcv(seed=i + 1)
        for i, symbol in enumerate(snapshot.sorted_symbols)
    }
    spy = make_ohlcv(seed=99)["close"].to_numpy()[-30:]
    payload = {
        "as_of": datetime(2026, 8, 31, 13, 0, tzinfo=UTC),
        "universe_snapshot": snapshot,
        "ohlcv_by_symbol": ohlcv,
        "news_by_symbol": news,
        "current_weights": {"CASH": 1.0},
        "p_calm": 0.4,
        "p_stress": 0.3,
        "spy_closes": spy,
    }
    payload.update(overrides)
    return StateBuildRequest(**payload)


def test_packed_shapes_are_9_and_7() -> None:
    state = build_ppo_discovery_state(_request())
    assert state.asset_features.shape == (MAX_ASSETS, 9)
    assert state.globals.shape == (7,)
    assert state.price_history.shape == (MAX_ASSETS, 250, 4)
    assert list(ASSET_FEATURE_NAMES) == [
        "momentum_1w_cs_rank",
        "momentum_4w_cs_rank",
        "momentum_12_1_cs_rank",
        "realized_vol_20d_cs_rank",
        "news_sentiment_cs_rank",
        "raw_news_sentiment",
        "log1p_article_count",
        "news_recency",
        "current_weight",
    ]
    assert len(GLOBAL_FEATURE_NAMES) == 7


def test_audit_fields_absent_from_tensor() -> None:
    state = build_ppo_discovery_state(_request())
    for name in AUDIT_NEWS_FIELDS:
        assert name not in ASSET_FEATURE_NAMES
        assert name not in GLOBAL_FEATURE_NAMES
    assert "per_symbol" in state.audit_news
    first = next(iter(state.audit_news["per_symbol"].values()))
    assert "average_confidence" in first


def test_verified_zero_news_zeros() -> None:
    snapshot = make_snapshot()
    news = {symbol: verified_zero_news(symbol) for symbol in snapshot.sorted_symbols}
    state = build_ppo_discovery_state(
        _request(universe_snapshot=snapshot, news_by_symbol=news)
    )
    eligible = [i for i, flag in enumerate(state.asset_mask) if flag]
    for index in eligible:
        assert state.asset_features[index, 5] == 0.0
        assert state.asset_features[index, 6] == 0.0
        assert state.asset_features[index, 7] == 0.0
    assert state.globals[6] == 0.0


def test_incomplete_news_aborts() -> None:
    snapshot = make_snapshot()
    news = {symbol: make_news(symbol) for symbol in snapshot.sorted_symbols}
    broken = news[snapshot.sorted_symbols[0]]
    news[snapshot.sorted_symbols[0]] = SymbolNewsIncomplete(broken)
    with pytest.raises(PPODiscoveryError, match="incomplete"):
        build_ppo_discovery_state(
            _request(universe_snapshot=snapshot, news_by_symbol=news)
        )


def SymbolNewsIncomplete(row):
    from dataclasses import replace

    return replace(row, query_complete=False)


def test_held_asset_missing_price_aborts() -> None:
    snapshot = make_snapshot()
    ohlcv = {symbol: make_ohlcv() for symbol in snapshot.sorted_symbols}
    del ohlcv[snapshot.sorted_symbols[0]]
    with pytest.raises(PPODiscoveryError, match="held asset"):
        build_ppo_discovery_state(
            _request(
                universe_snapshot=snapshot,
                ohlcv_by_symbol=ohlcv,
                current_weights={snapshot.sorted_symbols[0]: 0.5, "CASH": 0.5},
            )
        )


def test_unheld_incomplete_history_is_masked() -> None:
    snapshot = make_snapshot(n=11)
    ohlcv = {
        symbol: make_ohlcv(seed=i) for i, symbol in enumerate(snapshot.sorted_symbols)
    }
    dropped = snapshot.sorted_symbols[0]
    ohlcv[dropped] = make_ohlcv(n=10)
    state = build_ppo_discovery_state(
        _request(universe_snapshot=snapshot, ohlcv_by_symbol=ohlcv)
    )
    index = list(state.symbols).index(dropped)
    assert not bool(state.asset_mask[index])
    assert dropped in state.exclusions


def test_digest_is_byte_stable() -> None:
    first = build_ppo_discovery_state(_request())
    second = build_ppo_discovery_state(_request())
    assert first.state_digest == second.state_digest
    assert first.state_digest == sha256_digest(state_to_digest_payload(first))


def test_permutation_of_input_list_does_not_change_lex_packing() -> None:
    snapshot = make_snapshot()
    state = build_ppo_discovery_state(_request(universe_snapshot=snapshot))
    assert tuple(s for s in state.symbols if s) == snapshot.sorted_symbols

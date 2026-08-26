"""Dataset identity hashes bind OHLCV and news contents, not just dates."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from brain_api.core.ppo_discovery.dataset_identity import build_dataset_identity
from brain_api.core.ppo_discovery.environment import WeeklyTransition
from tests.core.ppo_discovery.factories import make_news, make_ohlcv, make_snapshot


def _week(snapshot, ohlcv, spy, *, sentiment: float) -> WeeklyTransition:
    index = spy.index
    return WeeklyTransition(
        cutoff=datetime(2020, 10, 16, 20, 0, tzinfo=UTC),
        rebalance_session=index[-5],
        next_rebalance_session=index[-1],
        news_by_symbol={
            symbol: make_news(symbol, raw=sentiment)
            for symbol in snapshot.sorted_symbols
        },
        p_calm=0.4,
        p_stress=0.3,
    )


def _frames(snapshot):
    index = pd.bdate_range("2019-01-02", periods=len(make_ohlcv()))
    ohlcv = {}
    for i, symbol in enumerate(snapshot.sorted_symbols):
        frame = make_ohlcv(seed=i + 1)
        frame.index = index
        ohlcv[symbol] = frame
    spy = make_ohlcv(seed=99)
    spy.index = index
    return ohlcv, spy


def test_dataset_hash_changes_when_close_or_news_changes() -> None:
    snapshot = make_snapshot(n=10)
    ohlcv, spy = _frames(snapshot)
    week = _week(snapshot, ohlcv, spy, sentiment=0.1)
    identity = build_dataset_identity(
        [week], [week], [week], snapshot=snapshot, ohlcv=ohlcv, spy=spy
    )
    ohlcv["S00"] = ohlcv["S00"].copy()
    ohlcv["S00"].iloc[-1, ohlcv["S00"].columns.get_loc("close")] += 1.0
    mutated_price = build_dataset_identity(
        [week], [week], [week], snapshot=snapshot, ohlcv=ohlcv, spy=spy
    )
    assert mutated_price.training_dataset_hash != identity.training_dataset_hash
    mutated_news_week = _week(snapshot, ohlcv, spy, sentiment=0.9)
    mutated_news = build_dataset_identity(
        [mutated_news_week],
        [mutated_news_week],
        [mutated_news_week],
        snapshot=snapshot,
        ohlcv=ohlcv,
        spy=spy,
    )
    assert mutated_news.evaluation_dataset_hash != identity.evaluation_dataset_hash

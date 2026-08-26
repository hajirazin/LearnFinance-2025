"""Closed-loop rollout: reward is computed from the sampled action."""

from __future__ import annotations

from datetime import UTC, datetime

import exchange_calendars as xcals
import pandas as pd
import torch

from brain_api.core.ppo_discovery.config import HISTORY_BARS, PPODiscoveryConfig
from brain_api.core.ppo_discovery.environment import (
    WeeklyTransition,
    collect_closed_loop_rollout,
)
from brain_api.core.ppo_discovery.policy import PPODiscoveryActorCritic
from brain_api.core.ppo_discovery.rewards import ppo_discovery_reward
from tests.core.ppo_discovery.factories import make_news, make_ohlcv, make_snapshot


def _xnys_index(n: int) -> pd.DatetimeIndex:
    calendar = xcals.get_calendar("XNYS")
    sessions = calendar.sessions_in_range(
        pd.Timestamp("2024-01-02"), pd.Timestamp("2026-08-01")
    )
    return pd.DatetimeIndex(sessions[-n:]).tz_localize(None).normalize()


def test_closed_loop_reward_matches_sampled_weights() -> None:
    index = _xnys_index(HISTORY_BARS + 10)
    snapshot = make_snapshot(11)
    ohlcv = {}
    for i, symbol in enumerate(snapshot.sorted_symbols):
        frame = make_ohlcv(n=len(index), seed=i + 1)
        frame.index = index
        ohlcv[symbol] = frame
    spy = make_ohlcv(n=len(index), seed=99)
    spy.index = index
    cutoff = datetime.combine(index[-6].date(), datetime.min.time(), tzinfo=UTC)
    news = {
        symbol: make_news(symbol, count=0, raw=0.0)
        for symbol in snapshot.sorted_symbols
    }
    week = WeeklyTransition(
        cutoff=cutoff.replace(hour=20),
        rebalance_session=index[-5],
        next_rebalance_session=index[-1],
        news_by_symbol=news,
        p_calm=0.4,
        p_stress=0.2,
    )
    config = PPODiscoveryConfig(dropout=0.0, training_nav_usd=100_000.0)
    policy = PPODiscoveryActorCritic(config)
    torch.manual_seed(0)
    steps = collect_closed_loop_rollout(
        policy,
        [week],
        snapshot=snapshot,
        ohlcv_by_symbol=ohlcv,
        spy=spy,
        feature_scalers={"log1p_article_count": {"mean": 0.0, "scale": 1.0}},
        config=config,
    )
    assert len(steps) == 1
    expected, _gross, _cost = ppo_discovery_reward(
        prior_weights={"CASH": 1.0},
        target_weights=steps[0].action.percentage_weights,
        symbol_returns=_returns_for(
            {"CASH": 1.0}, steps[0].action.percentage_weights, ohlcv, week
        ),
        symbol_prices=_prices_for(
            {"CASH": 1.0}, steps[0].action.percentage_weights, ohlcv, week
        ),
        nav_usd=config.training_nav_usd,
        config=config,
    )
    assert steps[0].reward == expected


def _returns_for(prior, weights, ohlcv, week):
    from brain_api.core.ppo_discovery.weeks import open_to_open_return

    out = {}
    names = set(prior) | set(weights)
    for symbol in names:
        if symbol == "CASH":
            continue
        if (
            abs(float(prior.get(symbol, 0.0))) <= 1e-12
            and abs(float(weights.get(symbol, 0.0))) <= 1e-12
        ):
            continue
        _open, simple = open_to_open_return(
            ohlcv[symbol],
            week.rebalance_session,
            week.next_rebalance_session,
            symbol=symbol,
        )
        out[symbol] = simple
    return out


def _prices_for(prior, weights, ohlcv, week):
    from brain_api.core.ppo_discovery.weeks import open_to_open_return

    out = {}
    names = set(prior) | set(weights)
    for symbol in names:
        if symbol == "CASH":
            continue
        if (
            abs(float(prior.get(symbol, 0.0))) <= 1e-12
            and abs(float(weights.get(symbol, 0.0))) <= 1e-12
        ):
            continue
        start_open, _simple = open_to_open_return(
            ohlcv[symbol],
            week.rebalance_session,
            week.next_rebalance_session,
            symbol=symbol,
        )
        out[symbol] = start_open
    return out

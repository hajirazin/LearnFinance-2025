"""Closed-loop rollout: reward is computed from the sampled action."""

from __future__ import annotations

from datetime import UTC, datetime

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest
import torch

from brain_api.core.portfolio_rl.broker_costs import (
    IBKRSingaporeCostConfig,
    compute_ibkr_rebalance_cost,
)
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
    config = PPODiscoveryConfig(dropout=0.0)
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
    expected, _gross, _cost, economic = ppo_discovery_reward(
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
    assert steps[0].realized_net_return == economic


def test_missing_unheld_price_frame_is_masked_not_aborted() -> None:
    index = _xnys_index(HISTORY_BARS + 10)
    snapshot = make_snapshot(11)
    ohlcv = {}
    for i, symbol in enumerate(snapshot.sorted_symbols):
        frame = make_ohlcv(n=len(index), seed=i + 1)
        frame.index = index
        ohlcv[symbol] = frame
    dropped = snapshot.sorted_symbols[0]
    ohlcv[dropped] = pd.DataFrame()
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
    config = PPODiscoveryConfig(dropout=0.0)
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
    dropped_index = list(steps[0].state.symbols).index(dropped)
    assert not bool(steps[0].state.asset_mask[dropped_index])


def test_reward_uses_locked_ibkr_costs_at_ten_thousand_dollars() -> None:
    config = PPODiscoveryConfig(hhi_penalty_scale=0.0)
    prior = {"AAPL": 0.10, "MSFT": 0.10, "CASH": 0.80}
    target = {"AAPL": 0.30, "MSFT": 0.30, "CASH": 0.40}
    prices = {"AAPL": 200.0, "MSFT": 100.0}

    reward, gross, cost_fraction, economic = ppo_discovery_reward(
        prior_weights=prior,
        target_weights=target,
        symbol_returns={"AAPL": 0.0, "MSFT": 0.0},
        symbol_prices=prices,
        nav_usd=config.training_nav_usd,
        config=config,
    )
    expected = compute_ibkr_rebalance_cost(
        symbol_order=["AAPL", "MSFT"],
        current_weights=np.array([0.10, 0.10, 0.80]),
        target_weights=np.array([0.30, 0.30, 0.40]),
        prices=np.array([200.0, 100.0]),
        cfg=IBKRSingaporeCostConfig.default(),
    )

    assert config.training_nav_usd == 10_000.0
    assert gross == 0.0
    assert cost_fraction == pytest.approx(expected.total_fraction)
    assert [leg.commission for leg in expected.legs] == pytest.approx([0.35, 0.35])
    assert economic == pytest.approx(np.log1p(-expected.total_fraction))
    assert reward == pytest.approx(economic)


def test_no_transaction_cost_ablation_is_zero_cost() -> None:
    _reward, _gross, cost_fraction, _economic = ppo_discovery_reward(
        prior_weights={"CASH": 1.0},
        target_weights={"AAPL": 0.98, "CASH": 0.02},
        symbol_returns={"AAPL": 0.0},
        symbol_prices={"AAPL": 100.0},
        nav_usd=10_000.0,
        config=PPODiscoveryConfig(),
        include_transaction_cost=False,
    )

    assert cost_fraction == 0.0


def test_reward_overlays_actual_nav_on_locked_ibkr_schedule() -> None:
    prior = {"AAPL": 0.10, "CASH": 0.90}
    target = {"AAPL": 0.30, "CASH": 0.70}
    symbol_order = ["AAPL"]
    current = np.array([0.10, 0.90])
    desired = np.array([0.30, 0.70])
    prices = np.array([200.0])

    _reward, _gross, cost_fraction, _economic = ppo_discovery_reward(
        prior_weights=prior,
        target_weights=target,
        symbol_returns={"AAPL": 0.0},
        symbol_prices={"AAPL": 200.0},
        nav_usd=1_000.0,
        config=PPODiscoveryConfig(hhi_penalty_scale=0.0),
    )
    expected_at_actual_nav = compute_ibkr_rebalance_cost(
        symbol_order=symbol_order,
        current_weights=current,
        target_weights=desired,
        prices=prices,
        cfg=IBKRSingaporeCostConfig.default().with_nav(1_000.0),
    )
    wrong_at_training_nav = compute_ibkr_rebalance_cost(
        symbol_order=symbol_order,
        current_weights=current,
        target_weights=desired,
        prices=prices,
        cfg=IBKRSingaporeCostConfig.default(),
    )

    assert cost_fraction == pytest.approx(expected_at_actual_nav.total_fraction)
    assert cost_fraction != pytest.approx(wrong_at_training_nav.total_fraction)


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

"""Closed-loop weekly environment: sample action, then Alpaca net reward."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
import torch

from brain_api.core.portfolio_rl.rewards import RebalanceTransition
from brain_api.core.ppo_discovery.config import PPODiscoveryConfig
from brain_api.core.ppo_discovery.policy import PPODiscoveryActorCritic
from brain_api.core.ppo_discovery.rewards import ppo_discovery_reward
from brain_api.core.ppo_discovery.rollout import RolloutStep
from brain_api.core.ppo_discovery.schemas import (
    CanonicalPPOState,
    PPODiscoveryError,
    SymbolNewsFeatures,
    UniverseSnapshot,
)
from brain_api.core.ppo_discovery.state_builder import (
    StateBuildRequest,
    build_ppo_discovery_state,
)
from brain_api.core.ppo_discovery.weeks import open_to_open_return, prices_as_of
from brain_api.core.sac.experience_accounting import build_rebalance_arrays


@dataclass(frozen=True)
class WeeklyTransition:
    """One decision week plus the next-open reward window."""

    cutoff: datetime
    rebalance_session: pd.Timestamp
    next_rebalance_session: pd.Timestamp
    news_by_symbol: Mapping[str, SymbolNewsFeatures]
    p_calm: float
    p_stress: float


def _spy_closes(spy: pd.DataFrame, cutoff: date) -> np.ndarray:
    sliced = prices_as_of(spy, cutoff)
    return sliced["close"].to_numpy(dtype=np.float64)


def collect_closed_loop_rollout(
    policy: PPODiscoveryActorCritic,
    transitions: Sequence[WeeklyTransition],
    *,
    snapshot: UniverseSnapshot,
    ohlcv_by_symbol: Mapping[str, pd.DataFrame],
    spy: pd.DataFrame,
    feature_scalers: Mapping[str, Any] | None,
    config: PPODiscoveryConfig,
    initial_weights: Mapping[str, float] | None = None,
    deterministic: bool = False,
    include_transaction_cost: bool = True,
    zero_news_features: bool = False,
    zero_hmm: bool = False,
    zero_history: bool = False,
    equal_weight_selected: bool = False,
    force_k: int | None = None,
) -> list[RolloutStep]:
    """Sample (or infer) an action, then reward it from next-open returns."""
    if not transitions:
        raise ValueError("closed-loop rollout requires at least one week")
    weights = dict(initial_weights or {"CASH": 1.0})
    steps: list[RolloutStep] = []
    policy.eval()
    for index, week in enumerate(transitions):
        cutoff_date = week.cutoff.date()
        ohlcv: dict[str, pd.DataFrame] = {}
        for symbol, frame in ohlcv_by_symbol.items():
            try:
                ohlcv[symbol] = prices_as_of(frame, cutoff_date)
            except PPODiscoveryError:
                continue
        news = dict(week.news_by_symbol)
        if zero_news_features:
            news = {
                symbol: SymbolNewsFeatures(
                    symbol=row.symbol,
                    raw_sentiment=0.0,
                    article_count=0,
                    average_confidence=row.average_confidence,
                    sentiment_dispersion=row.sentiment_dispersion,
                    hours_since_latest=0.0,
                    unique_source_count=row.unique_source_count,
                    has_news=0,
                    query_complete=row.query_complete,
                    news_recency=0.0,
                    log1p_article_count=0.0,
                    article_ids_sha256=row.article_ids_sha256,
                    request_manifest_sha256=row.request_manifest_sha256,
                )
                for symbol, row in news.items()
            }
        p_calm = 0.0 if zero_hmm else float(week.p_calm)
        p_stress = 0.0 if zero_hmm else float(week.p_stress)
        state = build_ppo_discovery_state(
            StateBuildRequest(
                as_of=week.cutoff,
                universe_snapshot=snapshot,
                ohlcv_by_symbol=ohlcv,
                news_by_symbol=news,
                current_weights=weights,
                p_calm=p_calm,
                p_stress=p_stress,
                spy_closes=_spy_closes(spy, cutoff_date),
                feature_scalers=feature_scalers,
            )
        )
        if zero_history:
            state.price_history[...] = 0.0
        prior = dict(weights)
        with torch.no_grad():
            if force_k is not None:
                target_weights, _order = policy.infer_decision(state, force_k=force_k)
                action = _action_from_weights(state, target_weights)
            elif deterministic or equal_weight_selected:
                target_weights = policy.infer_weights(state)
                action = _action_from_weights(state, target_weights)
                if equal_weight_selected:
                    action = _with_equal_stock_weights(action)
            else:
                action = policy.sample_action(state)
            value = float(policy.value(state).item())
            if deterministic or force_k is not None or equal_weight_selected:
                log_p = 0.0
            else:
                log_p = float(policy.log_prob(state, action).item())
        target = dict(action.percentage_weights)
        symbol_returns, symbol_prices = _next_open_market(
            prior,
            target,
            ohlcv_by_symbol,
            week.rebalance_session,
            week.next_rebalance_session,
        )
        reward, _gross, cost_fraction, economic_net_log = ppo_discovery_reward(
            prior_weights=prior,
            target_weights=target,
            symbol_returns=symbol_returns,
            symbol_prices=symbol_prices,
            nav_usd=config.training_nav_usd,
            config=config,
            include_transaction_cost=include_transaction_cost,
        )
        done = index == len(transitions) - 1
        steps.append(
            RolloutStep(
                state=state,
                action=action,
                reward=float(reward),
                value=value,
                log_p=log_p,
                done=done,
                realized_net_return=float(economic_net_log),
            )
        )
        weights = _post_rebalance_weights(
            target, symbol_returns, symbol_prices, cost_fraction
        )
    return steps


def _action_from_weights(
    state: CanonicalPPOState,
    weights: dict[str, float],
):
    from brain_api.core.ppo_discovery.schemas import SampledAction

    stocks = [
        symbol for symbol, weight in weights.items() if symbol != "CASH" and weight > 0
    ]
    stocks = sorted(stocks, key=lambda symbol: (-float(weights[symbol]), symbol))
    selected = tuple(stocks)
    return SampledAction(
        k=len(selected),
        selection_order=selected,
        selection_indices=tuple(state.symbols.index(symbol) for symbol in selected),
        z_cash=None if not selected else 0.0,
        dirichlet_weights=None
        if not selected
        else tuple(float(weights[symbol]) for symbol in selected),
        percentage_weights=dict(weights),
        log_p_k=0.0,
        log_p_selection=0.0,
        log_p_cash=0.0,
        log_p_dirichlet=0.0,
        log_p_total=0.0,
    )


def _with_equal_stock_weights(action):
    stocks = [symbol for symbol in action.percentage_weights if symbol != "CASH"]
    if not stocks:
        return action
    cash = float(action.percentage_weights["CASH"])
    mass = (1.0 - cash) / len(stocks)
    weights = dict.fromkeys(stocks, mass)
    weights["CASH"] = cash
    return action.__class__(**{**action.__dict__, "percentage_weights": weights})


def _next_open_market(
    prior: Mapping[str, float],
    target: Mapping[str, float],
    ohlcv_by_symbol: Mapping[str, pd.DataFrame],
    start_session: pd.Timestamp,
    end_session: pd.Timestamp,
) -> tuple[dict[str, float], dict[str, float]]:
    returns: dict[str, float] = {}
    prices: dict[str, float] = {}
    names = set(prior) | set(target)
    for symbol in names:
        if symbol == "CASH":
            continue
        prior_w = abs(float(prior.get(symbol, 0.0)))
        target_w = abs(float(target.get(symbol, 0.0)))
        if prior_w <= 1e-12 and target_w <= 1e-12:
            continue
        frame = ohlcv_by_symbol.get(symbol)
        if frame is None:
            raise ValueError(f"missing OHLCV for traded symbol {symbol}")
        start_open, simple = open_to_open_return(
            frame, start_session, end_session, symbol=symbol
        )
        returns[symbol] = simple
        prices[symbol] = start_open
    return returns, prices


def _post_rebalance_weights(
    target: Mapping[str, float],
    symbol_returns: Mapping[str, float],
    symbol_prices: Mapping[str, float],
    cost_fraction: float,
) -> dict[str, float]:
    """Next week's prior is the after-cost drifted portfolio, not the target."""
    symbol_order, _prior, target_arr, _prices = build_rebalance_arrays(
        dict(target), dict(target), dict(symbol_prices)
    )
    stock_returns = np.asarray(
        [float(symbol_returns.get(symbol, 0.0)) for symbol in symbol_order],
        dtype=np.float64,
    )
    transition = RebalanceTransition.calculate(target_arr, stock_returns, cost_fraction)
    next_weights = {
        symbol: float(weight)
        for symbol, weight in zip(
            symbol_order, transition.post_weights[:-1], strict=True
        )
    }
    next_weights["CASH"] = float(transition.post_weights[-1])
    return next_weights


__all__ = ["WeeklyTransition", "collect_closed_loop_rollout"]

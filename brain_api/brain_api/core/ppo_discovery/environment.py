"""Closed-loop weekly environment: sample action, then IBKR net reward."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd
import torch

from brain_api.core.ppo_discovery.config import PPODiscoveryConfig
from brain_api.core.ppo_discovery.policy import PPODiscoveryActorCritic
from brain_api.core.ppo_discovery.rewards import ppo_discovery_reward
from brain_api.core.ppo_discovery.rollout import RolloutStep
from brain_api.core.ppo_discovery.schemas import (
    CanonicalPPOState,
    SymbolNewsFeatures,
    UniverseSnapshot,
)
from brain_api.core.ppo_discovery.state_builder import (
    StateBuildRequest,
    build_ppo_discovery_state,
)
from brain_api.core.ppo_discovery.weeks import open_to_open_return, prices_as_of


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
    feature_scalers: Mapping[str, Mapping[str, float]] | None,
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
        ohlcv = {
            symbol: prices_as_of(frame, cutoff_date)
            for symbol, frame in ohlcv_by_symbol.items()
        }
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
        with torch.no_grad():
            if deterministic or force_k is not None or equal_weight_selected:
                weights = policy.infer_weights(state)
                action = _action_from_weights(state, weights, force_k=force_k)
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
            weights,
            target,
            ohlcv_by_symbol,
            week.rebalance_session,
            week.next_rebalance_session,
        )
        reward, _gross, _cost = ppo_discovery_reward(
            prior_weights=weights,
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
            )
        )
        weights = target
    return steps


def _action_from_weights(
    state: CanonicalPPOState,
    weights: dict[str, float],
    *,
    force_k: int | None = None,
):
    from brain_api.core.ppo_discovery.config import CASH_FLOOR
    from brain_api.core.ppo_discovery.schemas import SampledAction

    stocks = [
        symbol for symbol, weight in weights.items() if symbol != "CASH" and weight > 0
    ]
    stocks = sorted(stocks, key=lambda symbol: (-float(weights[symbol]), symbol))
    if force_k is not None:
        eligible = [
            symbol
            for symbol, flag in zip(
                state.symbols, state.asset_mask.tolist(), strict=True
            )
            if flag and symbol
        ]
        stocks = sorted(
            eligible,
            key=lambda symbol: (-float(weights.get(symbol, 0.0)), symbol),
        )[:force_k]
        if stocks:
            cash = max(float(weights.get("CASH", CASH_FLOOR)), CASH_FLOOR)
            mass = (1.0 - cash) / len(stocks)
            weights = dict.fromkeys(stocks, mass)
            weights["CASH"] = cash
        else:
            weights = {"CASH": 1.0}
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


__all__ = ["WeeklyTransition", "collect_closed_loop_rollout"]

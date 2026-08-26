"""IBKR after-cost reward plus the existing HHI penalty for ppo_discovery."""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace

import numpy as np

from brain_api.core.portfolio_rl.broker_costs import (
    IBKRSingaporeCostConfig,
    compute_ibkr_rebalance_cost,
)
from brain_api.core.portfolio_rl.rewards import compute_net_log_reward
from brain_api.core.ppo_discovery.config import PPODiscoveryConfig
from brain_api.core.ppo_discovery.schemas import PPODiscoveryError
from brain_api.core.sac.experience_accounting import build_rebalance_arrays


def ppo_discovery_reward(
    *,
    prior_weights: Mapping[str, float],
    target_weights: Mapping[str, float],
    symbol_returns: Mapping[str, float],
    symbol_prices: Mapping[str, float],
    nav_usd: float,
    config: PPODiscoveryConfig,
    include_transaction_cost: bool = True,
) -> tuple[float, float, float]:
    """Return ``(reward, gross_return, cost_fraction)``.

    Missing next-open prices for a traded name fail rather than last-price fill.
    ``n_stocks`` for HHI is the number of selected stocks ``K``, not 15.
    """
    stocks = [symbol for symbol in target_weights if symbol != "CASH"]
    for symbol in stocks:
        if abs(float(target_weights[symbol])) <= 1e-12:
            continue
        if symbol not in symbol_returns or not np.isfinite(symbol_returns[symbol]):
            raise PPODiscoveryError(
                f"missing next-open return for held/selected symbol {symbol}"
            )
    gross = 0.0
    for symbol, weight in target_weights.items():
        if symbol == "CASH" or abs(float(weight)) <= 1e-12:
            continue
        gross += float(weight) * float(symbol_returns[symbol])
    cost_config = IBKRSingaporeCostConfig.default().with_nav(nav_usd)
    symbol_order, prior, target, prices = build_rebalance_arrays(
        dict(prior_weights), dict(target_weights), dict(symbol_prices)
    )
    cost_fraction = 0.0
    if include_transaction_cost:
        cost = compute_ibkr_rebalance_cost(
            symbol_order=symbol_order,
            current_weights=prior,
            target_weights=target,
            prices=prices,
            cfg=cost_config,
        )
        cost_fraction = float(cost.total_fraction)
    k = len(stocks)
    stock_plus_cash = np.zeros(k + 1, dtype=np.float64)
    for index, symbol in enumerate(stocks):
        stock_plus_cash[index] = float(target_weights[symbol])
    stock_plus_cash[-1] = float(target_weights.get("CASH", 0.0))
    reward_cfg = SimpleNamespace(
        reward_scale=config.reward_scale,
        hhi_penalty_scale=config.hhi_penalty_scale,
        n_stocks=k,
    )
    reward = compute_net_log_reward(
        gross, cost_fraction, reward_cfg, target_weights=stock_plus_cash
    )
    return float(reward), float(gross), cost_fraction


__all__ = ["ppo_discovery_reward"]

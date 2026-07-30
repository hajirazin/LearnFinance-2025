"""Strict realized SAC accounting shared by experience-label endpoints."""

from __future__ import annotations

import numpy as np

from brain_api.core.portfolio_rl.broker_costs import (
    IBKRSingaporeCostConfig,
    compute_ibkr_rebalance_cost,
)
from brain_api.core.portfolio_rl.rewards import compute_exact_net_log_return


def build_rebalance_arrays(
    prior_weights: dict[str, float],
    target_weights: dict[str, float],
    symbol_prices: dict[str, float],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Align weights and require a usable price for every traded stock."""
    stock_symbols = sorted(
        {symbol for symbol in prior_weights if symbol != "CASH"}
        | {symbol for symbol in target_weights if symbol != "CASH"}
    )
    n_stocks = len(stock_symbols)
    prior = np.zeros(n_stocks + 1)
    target = np.zeros(n_stocks + 1)
    prices = np.zeros(n_stocks)
    for index, symbol in enumerate(stock_symbols):
        prior[index] = float(prior_weights.get(symbol, 0.0))
        target[index] = float(target_weights.get(symbol, 0.0))
        delta = abs(target[index] - prior[index])
        if delta > 1e-9:
            price = symbol_prices.get(symbol)
            if price is None or not np.isfinite(price) or price <= 0:
                raise ValueError(
                    f"price for {symbol!r} required to size the rebalance "
                    f"leg (delta_w={delta}); got price={price!r}"
                )
            prices[index] = float(price)
    prior[-1] = float(prior_weights.get("CASH", 0.0))
    target[-1] = float(target_weights.get("CASH", 0.0))
    return stock_symbols, prior, target, prices


def compute_realized_sac_reward(
    target_weights: dict[str, float],
    symbol_returns: dict[str, float],
    *,
    prior_weights: dict[str, float] | None = None,
    symbol_prices: dict[str, float] | None = None,
    nav_usd: float | None = None,
    reward_scale: float = 100.0,
) -> tuple[float, float]:
    """Compute exact net-log reward, rejecting unobserved held-symbol returns."""
    gross_return = 0.0
    for symbol, weight in target_weights.items():
        if symbol == "CASH" or abs(weight) <= 1e-12:
            continue
        if symbol not in symbol_returns:
            raise ValueError(f"Missing realized return for held symbol {symbol}")
        realized_return = float(symbol_returns[symbol])
        if not np.isfinite(realized_return):
            raise ValueError(f"Non-finite realized return for {symbol}")
        gross_return += float(weight) * realized_return

    prior_weights = prior_weights or {"CASH": 1.0}
    symbol_prices = symbol_prices or {}
    cost_config = IBKRSingaporeCostConfig.default()
    if nav_usd is not None:
        cost_config = cost_config.with_nav(nav_usd)
    symbol_order, prior, target, prices = build_rebalance_arrays(
        prior_weights, target_weights, symbol_prices
    )
    cost = compute_ibkr_rebalance_cost(
        symbol_order=symbol_order,
        current_weights=prior,
        target_weights=target,
        prices=prices,
        cfg=cost_config,
    )
    reward = (
        compute_exact_net_log_return(gross_return, cost.total_fraction) * reward_scale
    )
    return reward, gross_return

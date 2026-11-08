"""Self-financing fractional-share account with the configured IBKR fees."""

from dataclasses import dataclass, field
from math import isfinite

import numpy as np

from brain_api.core.portfolio_rl.broker_costs import (
    IBKRSingaporeCostConfig,
    compute_ibkr_leg_cost,
    compute_ibkr_rebalance_cost,
)
from brain_api.core.sac.experience_accounting import build_rebalance_arrays


@dataclass
class PortfolioBook:
    cash: float
    shares: dict[str, float] = field(default_factory=dict)

    def value(self, prices):
        if not isfinite(self.cash) or self.cash < -1e-7:
            raise ValueError("Cash must be finite and nonnegative")
        total = self.cash
        for symbol, quantity in self.shares.items():
            price = prices.get(symbol)
            if price is None or not isfinite(price) or price <= 0:
                raise ValueError(f"Missing positive mark for held asset {symbol}")
            if not isfinite(quantity) or quantity < 0:
                raise ValueError(f"Invalid quantity for {symbol}")
            total += quantity * price
        if not isfinite(total) or total <= 0:
            raise ValueError("Portfolio NAV must be finite and positive")
        return float(total)

    def weights(self, prices):
        nav = self.value(prices)
        return {
            **{s: q * prices[s] / nav for s, q in self.shares.items()},
            "CASH": self.cash / nav,
        }

    def rebalance(self, target_weights, prices, *, weight_epsilon=1e-9):
        if not all(isfinite(w) and w >= 0 for w in target_weights.values()):
            raise ValueError("Invalid target weights")
        if not np.isclose(sum(target_weights.values()), 1, atol=1e-10, rtol=0):
            raise ValueError("Target weights must sum to one")
        nav = self.value(prices)
        symbols, prior, target, aligned_prices = build_rebalance_arrays(
            self.weights(prices), target_weights, prices
        )
        costs = compute_ibkr_rebalance_cost(
            symbol_order=symbols,
            current_weights=prior,
            target_weights=target,
            prices=aligned_prices,
            cfg=IBKRSingaporeCostConfig.default().with_nav(nav),
            weight_epsilon=weight_epsilon,
        )
        quantities = {
            s: nav * w / prices[s]
            for s, w in target_weights.items()
            if s != "CASH" and w > 0
        }
        cash = nav * target_weights.get("CASH", 0) - costs.total_usd
        next_book = PortfolioBook(cash, quantities)
        trades = []
        for leg in costs.legs:
            before = self.shares.get(leg.symbol, 0)
            after = quantities.get(leg.symbol, 0)
            trades.append(
                {
                    "symbol": leg.symbol,
                    "side": leg.side,
                    "quantity_delta": after - before,
                    "price": float(prices[leg.symbol]),
                    "notional_usd": abs(after - before) * prices[leg.symbol],
                    **leg.to_dict(),
                }
            )
        # No external cash flow: sales fund purchases; fees reduce only cash.
        dollars_traded = sum(
            (quantities.get(s, 0) - self.shares.get(s, 0)) * prices[s]
            for s in set(self.shares) | set(quantities)
        )
        assert np.isclose(cash, self.cash - dollars_traded - costs.total_usd, atol=1e-7)
        assert np.isclose(next_book.value(prices), nav - costs.total_usd, atol=1e-7)
        return next_book, {
            "nav_before": nav,
            "nav_after": next_book.value(prices),
            "fees_usd": costs.total_usd,
            "fee_breakdown": costs.breakdown(),
            "trades": trades,
        }


def buy_and_hold_book(symbol, price, capital):
    """Buy as many fractional shares as capital permits, including entry fees."""
    if not isfinite(price) or price <= 0:
        raise ValueError("Benchmark entry price must be positive")
    config = IBKRSingaporeCostConfig.default().with_nav(capital)
    low, high = 0.0, capital
    for _ in range(70):
        notional = (low + high) / 2
        fee = compute_ibkr_leg_cost(
            symbol=symbol,
            notional_usd=notional,
            shares=notional / price,
            side="buy",
            cfg=config,
        ).total
        if notional + fee <= capital:
            low = notional
        else:
            high = notional
    fee = compute_ibkr_leg_cost(
        symbol=symbol, notional_usd=low, shares=low / price, side="buy", cfg=config
    ).total
    cash = capital - low - fee
    assert cash >= -1e-8
    return PortfolioBook(max(cash, 0), {symbol: low / price}), fee

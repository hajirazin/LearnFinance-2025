"""Fund IBKR fees from invested capital while preserving target proportions."""

import numpy as np
from hrp_replay import mark_to_market, positive_price, validate_weights
from scipy.optimize import brentq

from brain_api.core.portfolio_rl.broker_costs import (
    IBKRSingaporeCostConfig,
    compute_ibkr_rebalance_cost,
)


def fee_rebalance(shares, weights, opening_prices, cash, weight_epsilon):
    validate_weights(weights, weights)
    if not np.isfinite(cash) or cash < 0:
        raise ValueError("Cash must be finite and nonnegative")
    nav = cash + mark_to_market(shares, opening_prices)
    if not np.isfinite(nav) or nav <= 0:
        raise ValueError("Opening NAV must be positive")
    symbols = sorted(set(shares) | set(weights))
    price_array = np.array([positive_price(opening_prices, s) for s in symbols])
    prior = np.r_[
        [shares.get(s, 0) * opening_prices[s] / nav for s in symbols], cash / nav
    ]
    risky_target = np.array([weights.get(s, 0) for s in symbols])
    config = IBKRSingaporeCostConfig.default().with_nav(nav)

    def costs(invested):
        target = np.r_[risky_target * invested / nav, 1 - invested / nav]
        return compute_ibkr_rebalance_cost(
            symbols, prior, target, price_array, config, weight_epsilon=weight_epsilon
        )

    # Investing the full pre-fee NAV would leave cash negative. Solve for the
    # post-fee capital and charge the resulting actual buy/sell quantities.
    invested = brentq(
        lambda value: value + costs(value).total_usd - nav, 0, nav, xtol=1e-9
    )
    fees = costs(invested)
    target = {
        s: invested * w / positive_price(opening_prices, s) for s, w in weights.items()
    }
    after = mark_to_market(target, opening_prices)
    details = {leg.symbol: leg.to_dict() for leg in fees.legs}
    trades = []
    for s in symbols:
        delta = target.get(s, 0) - shares.get(s, 0)
        notional = abs(delta) * opening_prices[s]
        if notional <= 1e-8:
            continue
        cost = details.get(
            s,
            dict.fromkeys(
                [
                    "commission",
                    "sec_fee",
                    "finra_taf",
                    "finra_cat",
                    "clearing",
                    "pass_through",
                    "total",
                ],
                0.0,
            ),
        )
        trades.append(
            {
                **cost,
                "symbol": s,
                "side": "buy" if delta > 0 else "sell",
                "quantity_before": shares.get(s, 0),
                "quantity_after": target.get(s, 0),
                "quantity_delta": delta,
                "price": opening_prices[s],
                "shares": abs(delta),
                "notional_usd": notional,
                "fee_exempt": s not in details,
            }
        )
    cash_after = (
        cash - sum(t["quantity_delta"] * t["price"] for t in trades) - fees.total_usd
    )
    if not np.isclose(cash_after, 0, rtol=0, atol=1e-7):
        raise ArithmeticError("Fee-funded rebalance did not conserve cash")
    if not np.isclose(after + fees.total_usd, nav, rtol=0, atol=1e-7):
        raise ArithmeticError("Fee-funded rebalance did not conserve NAV")
    return target, {
        "nav_before": nav,
        "nav_after": after,
        "fees_usd": fees.total_usd,
        "fee_breakdown": fees.breakdown(),
        "cash_after": cash_after,
        "weight_epsilon": weight_epsilon,
        "trades": trades,
    }

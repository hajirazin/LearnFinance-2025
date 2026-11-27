"""Zero-cost, fractional-share accounting with holdings carried overnight."""

from math import isfinite

import numpy as np
import pandas as pd


def positive_price(prices, symbol):
    price = prices.get(symbol)
    if price is None or not isfinite(price) or price <= 0:
        raise ValueError(f"Missing finite positive price for {symbol}: {price}")
    return float(price)


def mark_to_market(shares, prices):
    if any(not isfinite(q) or q < 0 for q in shares.values()):
        raise ValueError("Invalid share quantity")
    return float(sum(q * positive_price(prices, s) for s, q in shares.items()))


def equal_weight_rebalance(shares, selected, opening_prices, cash):
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Selections must be nonempty and unique")
    if not isfinite(cash) or cash < 0:
        raise ValueError("Invalid cash")
    nav = cash + mark_to_market(shares, opening_prices)
    if not isfinite(nav) or nav <= 0:
        raise ValueError("NAV must be positive")
    target = {
        s: nav / len(selected) / positive_price(opening_prices, s) for s in selected
    }
    trades = [
        {
            "symbol": s,
            "quantity_before": shares.get(s, 0),
            "quantity_after": target.get(s, 0),
            "quantity_delta": target.get(s, 0) - shares.get(s, 0),
            "price": positive_price(opening_prices, s),
        }
        for s in sorted(set(shares) | set(target))
    ]
    dollars = sum(t["quantity_delta"] * t["price"] for t in trades)
    if not np.isclose(dollars, cash, rtol=0, atol=1e-7):
        raise ArithmeticError("Rebalance is not self-financing")
    after = mark_to_market(target, opening_prices)
    if not np.isclose(after, nav, rtol=0, atol=1e-7):
        raise ArithmeticError("Rebalance changed NAV")
    return target, {
        "nav_before": nav,
        "nav_after": after,
        "cash_after": 0,
        "fees": 0,
        "trades": trades,
        "gross_turnover": sum(abs(t["quantity_delta"] * t["price"]) for t in trades)
        / nav,
    }


def weekly_schedule(sessions, start, end):
    """Holiday Mondays trade Tuesday; Friday holidays mark the last weekly session."""
    rows = []
    for monday in pd.date_range(start, end, freq="W-MON"):
        friday = monday + pd.Timedelta(days=4)
        if friday.date() > end:
            continue
        week = sessions[(sessions >= monday) & (sessions <= friday)]
        previous = sessions[sessions < monday]
        if week.empty or previous.empty:
            raise ValueError(f"Calendar evidence missing for {monday}")
        rows.append(
            {
                "monday": monday,
                "anchor": previous[-1],
                "execution": week[0],
                "valuation": week[-1],
            }
        )
    return rows


def completed_weekly_schedule(closes, start, end, as_of):
    """Retain a final week-to-date point, never an unfinished daily candle."""
    complete = closes.loc[closes <= pd.Timestamp(as_of)]
    rows = weekly_schedule(complete.index, start, end)
    for row in rows:
        friday = row["monday"] + pd.Timedelta(days=4)
        week = closes.loc[row["monday"] : friday]
        row["week_complete"] = row["valuation"] == week.index[-1]
    return rows

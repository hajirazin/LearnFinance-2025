"""HRP targets and self-financing accounting on frozen PatchTST selections."""

import numpy as np
import pandas as pd
from scratch.patchtst_2025_snapshot_2026_replay.accounting import (
    mark_to_market,
    positive_price,
)

from brain_api.core.hrp import compute_hrp_weights

LABEL = "PatchTST + HRP"
INITIAL = 10000.0
LOOKBACK = 252


def validate_weights(weights, selected):
    values = np.array(list(weights.values()), dtype=float)
    if (
        not weights
        or set(weights) != set(selected)
        or not np.isfinite(values).all()
        or (values <= 0).any()
        or not np.isclose(values.sum(), 1, rtol=0, atol=1e-12)
    ):
        raise ValueError(
            "HRP weights must allocate one positive simplex to every selection"
        )


def hrp_target(prices, selected, sessions, anchor, lookback=LOOKBACK):
    """Exactly lookback adjusted simple returns, ending before the decision."""
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Selections must be nonempty and unique")
    dates = sessions[sessions <= anchor][-(lookback + 1) :]
    if len(dates) != lookback + 1 or dates[-1] != anchor:
        raise ValueError("Insufficient exchange-session history at forecast anchor")
    history = pd.DataFrame(
        {symbol: prices[symbol]["close"].reindex(dates) for symbol in selected}
    )
    values = history.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(
            "Selected-stock HRP history must be complete, finite and positive"
        )
    returns = history.pct_change(fill_method=None).iloc[1:]
    if (returns.std() <= 0).any():
        raise ValueError("Selected-stock HRP history has zero return variance")
    weights = compute_hrp_weights(returns)
    validate_weights(weights, selected)
    return weights, {
        "history_start": dates[0],
        "history_end": dates[-1],
        "daily_returns": len(returns),
        "returns_sha256": __import__("hashlib")
        .sha256(returns.to_csv(float_format="%.17g").encode())
        .hexdigest(),
    }


def weighted_rebalance(shares, weights, opening_prices, cash):
    """Fully invest the current NAV; sales finance buys, with no new deposit."""
    validate_weights(weights, weights)
    if not np.isfinite(cash) or cash < 0:
        raise ValueError("Invalid cash")
    nav = cash + mark_to_market(shares, opening_prices)
    if not np.isfinite(nav) or nav <= 0:
        raise ValueError("NAV must be positive")
    target = {
        s: nav * weight / positive_price(opening_prices, s)
        for s, weight in weights.items()
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
    after = mark_to_market(target, opening_prices)
    if not np.isclose(dollars, cash, rtol=0, atol=1e-7):
        raise ArithmeticError("Rebalance is not self-financing")
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


def quotes(prices, symbols, stamp, field):
    return {s: positive_price({s: prices[s].at[stamp, field]}, s) for s in symbols}


def simulate_hrp(prices, plan, sessions, lookback=LOOKBACK):
    shares, ledger, holdings, trades = {}, [], [], []
    navs, marks = [INITIAL], [plan[0]["execution"]]
    for week in plan:
        if (
            not week["anchor"]
            < week["monday"]
            <= week["execution"]
            <= week["valuation"]
        ):
            raise ValueError(
                "Prediction anchor must precede the decision and execution"
            )
        weights, evidence = hrp_target(
            prices, week["selected"], sessions, week["anchor"], lookback
        )
        opening = quotes(prices, set(shares) | set(weights), week["execution"], "open")
        shares, rebalance = weighted_rebalance(
            shares, weights, opening, INITIAL if not ledger else 0
        )
        closing = quotes(prices, shares, week["valuation"], "close")
        nav = mark_to_market(shares, closing)
        ledger.append(
            {
                **week,
                "history": evidence,
                "target_weights": weights,
                "prior_valuation_nav": navs[-1],
                "weekly_return": nav / navs[-1] - 1,
                "weekend_return": rebalance["nav_before"] / navs[-1] - 1,
                "friday_close_nav": nav,
                "rebalance": rebalance,
            }
        )
        for symbol, quantity in shares.items():
            holdings.append(
                {
                    "monday": week["monday"],
                    "execution": week["execution"],
                    "valuation": week["valuation"],
                    "symbol": symbol,
                    "target_weight": weights[symbol],
                    "quantity_adjusted_units": quantity,
                    "opening_price": opening[symbol],
                    "closing_price": closing[symbol],
                    "opening_value": quantity * opening[symbol],
                    "closing_value": quantity * closing[symbol],
                    "closing_weight": quantity * closing[symbol] / nav,
                }
            )
        trades.extend(
            {"monday": week["monday"], "execution": week["execution"], **trade}
            for trade in rebalance["trades"]
        )
        navs.append(nav)
        marks.append(week["valuation"])
    return (
        pd.Series(navs, index=pd.DatetimeIndex(marks, name="date"), name=LABEL),
        ledger,
        pd.DataFrame(holdings),
        pd.DataFrame(trades),
    )

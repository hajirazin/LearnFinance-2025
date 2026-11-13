"""Apply the production selector and portfolio weights to next-open simulated fills."""

import math
from datetime import timedelta

import numpy as np
import pandas as pd
from brain_api.core.hrp import compute_hrp_allocation
from brain_api.core.sticky_selection import select_with_rank_band

INITIAL_NAV = 10_000.0
SPREAD_SLIPPAGE_BPS = (0, 5, 10, 25)
SEC_RATE = 0.00002060
TAF_RATE = 0.000195
TAF_CAP = 9.79
CAT_RATE = 0.000003


def allocation_path(scores, closes, *, equal_weight=False):
    previous = None
    allocations = []
    for friday, week in scores.groupby(level="friday", sort=True):
        selection = select_with_rank_band(
            week.droplevel("friday").to_dict(), previous, 15, 20
        )
        if equal_weight:
            if len(selection.selected) != 15:
                raise ValueError(f"Expected 15 selections on {friday.date()}")
            weights = pd.Series(1 / 15, index=selection.selected, dtype=float)
            previous = set(selection.selected)
            allocations.append((friday, weights, selection))
            continue
        history = closes.loc[friday - timedelta(days=423) : friday]
        result = compute_hrp_allocation(
            {
                symbol: history[[symbol]].rename(columns={symbol: "close"})
                for symbol in selection.selected
            },
            lookback_days=252,
            as_of_date=friday.date(),
        )
        if len(result.percentage_weights) != 15:
            raise ValueError(
                f"HRP produced {len(result.percentage_weights)} names on {friday.date()}"
            )
        weights = pd.Series(result.percentage_weights, dtype=float)
        if not np.isfinite(weights).all() or not weights.gt(0).all():
            raise ValueError(f"Invalid HRP weights on {friday.date()}")
        weights /= weights.sum()
        previous = set(weights.index)
        allocations.append((friday, weights, selection))
    return allocations


def transaction_fees(delta, raw_open, bps):
    sold = -delta[delta.lt(-1e-9)]
    sec = sum(math.ceil(value * SEC_RATE * 100) / 100 for value in sold)
    taf = sum(
        math.ceil(min(TAF_CAP, value * TAF_RATE) * 100) / 100
        for value in sold / raw_open[sold.index]
    )
    cat = (
        math.ceil(float((delta.abs() / raw_open[delta.index]).sum()) * CAT_RATE * 100)
        / 100
    )
    return float(delta.abs().sum()) * bps / 10_000, sec + taf + cat


def rebalance(current, weights, raw_open, cash, bps):
    """Solve the cash budget after fees while preserving target proportions."""
    symbols = current.index.union(weights.index).sort_values()
    current = current.reindex(symbols, fill_value=0)
    weights = weights.reindex(symbols, fill_value=0)
    nav = float(current.sum()) + cash

    def trades(invested):
        delta = invested * weights - current
        friction, regulatory = transaction_fees(delta, raw_open, bps)
        return delta, friction, regulatory

    low, high = 0.0, nav
    for _ in range(45):
        mid = (low + high) / 2
        _, friction, regulatory = trades(mid)
        if mid + friction + regulatory <= nav:
            low = mid
        else:
            high = mid
    delta, friction, regulatory = trades(low)
    remaining_cash = nav - low - friction - regulatory
    if remaining_cash < -1e-8:
        raise ValueError("Negative cash after rebalancing")
    return (
        low * weights[weights.gt(0)],
        delta,
        friction,
        regulatory,
        max(remaining_cash, 0),
    )


def simulate(allocations, opens, raw_opens, closes, terminal, bps):
    units = pd.Series(dtype=float)
    cash = INITIAL_NAV
    ledger, daily, orders = [], [], []
    executions = [
        opens.index[opens.index.searchsorted(friday, side="right")]
        for friday, _, _ in allocations
    ]
    if len(set(executions)) != len(executions) or not executions[-1] < terminal:
        raise ValueError("Invalid execution schedule")
    for (friday, weights, _selection), fill, end in zip(
        allocations, executions, [*executions[1:], terminal], strict=True
    ):
        traded = units.index.union(weights.index)
        execution_prices = opens.loc[fill, traded]
        raw = raw_opens.loc[fill, traded]
        if not (
            np.isfinite(execution_prices).all()
            and np.isfinite(raw).all()
            and execution_prices.gt(0).all()
            and raw.gt(0).all()
        ):
            raise ValueError(f"Missing or invalid execution price at {fill.date()}")
        current = units * execution_prices.reindex(units.index)
        start_nav = float(current.sum()) + cash
        invested, delta, friction, regulatory, cash = rebalance(
            current, weights, raw, cash, bps
        )
        units = invested / execution_prices[invested.index]
        marks = closes.loc[(closes.index >= fill) & (closes.index < end), units.index]
        end_prices = opens.loc[end, units.index]
        if not np.isfinite(marks).all().all() or not np.isfinite(end_prices).all():
            raise ValueError(
                f"Missing held price between {fill.date()} and {end.date()}"
            )
        end_nav = float(units.dot(end_prices)) + cash
        daily.append({"date": fill, "mark": "open_before_trade", "nav": start_nav})
        daily.append(
            {
                "date": fill,
                "mark": "open_after_trade",
                "nav": float(invested.sum()) + cash,
            }
        )
        daily.extend(
            {"date": day, "mark": "close", "nav": value}
            for day, value in (marks @ units + cash).items()
        )
        ledger.append(
            {
                "friday": friday,
                "fill": fill,
                "next_mark": end,
                "pretrade_nav": start_nav,
                "end_nav": end_nav,
                "net_return": end_nav / start_nav - 1,
                "spread_slippage": friction,
                "regulatory_fees": regulatory,
                "traded_notional": float(delta.abs().sum()),
                "trades": int(delta.abs().gt(1e-9).sum()),
            }
        )
        orders.extend(
            {
                "fill": fill,
                "symbol": symbol,
                "signed_notional": value,
                "raw_open": raw[symbol],
                "absolute_shares": abs(value) / raw[symbol],
            }
            for symbol, value in delta.items()
            if abs(value) > 1e-9
        )
    daily.append({"date": terminal, "mark": "terminal_open", "nav": end_nav})
    daily_frame = pd.DataFrame(daily)
    nav = daily_frame["nav"]
    metrics = {
        "net_return": end_nav / INITIAL_NAV - 1,
        "maximum_drawdown": (nav / nav.cummax() - 1).min(),
        "final_nav": end_nav,
        "cost_usd": sum(
            row["spread_slippage"] + row["regulatory_fees"] for row in ledger
        ),
        "total_traded_notional": sum(row["traded_notional"] for row in ledger),
    }
    return metrics, pd.DataFrame(ledger), daily_frame, pd.DataFrame(orders)

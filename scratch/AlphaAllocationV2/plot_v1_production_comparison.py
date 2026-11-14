"""Plot broker equity, SPY, and a weekly hindsight top-15 portfolio."""

import json
from itertools import pairwise

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from build_friday_dataset import plt
from compare_patchtst import write_json
from fetch_v1_production_comparison import NY, OUTPUT, START
from matplotlib.dates import ConciseDateFormatter, MonthLocator
from matplotlib.ticker import PercentFormatter

LABELS = {
    "v1": "V1 · Alpaca paper account",
    "spy": "SPY · buy and hold",
    "ideal": "Hindsight · equal-weight top 15 before costs",
}
COLORS = {"v1": "#315c88", "spy": "#4a7754", "ideal": "#aa602c"}


def broker_equity(evidence, sessions, calendar):
    history = evidence["history"]
    index = pd.to_datetime(history["timestamp"], unit="s", utc=True)
    days = index.tz_convert(NY).tz_localize(None).normalize()
    equity = pd.Series(history["equity"], index=days, dtype=float)
    source = "Alpaca daily equity"
    end = sessions[-1]
    if end not in equity.index:
        intraday = evidence["latest_intraday"]
        timestamps = pd.to_datetime(intraday["timestamp"], unit="s", utc=True)
        closing = calendar.session_close(end)
        equity.loc[end] = pd.Series(intraday["equity"], index=timestamps).loc[closing]
        source = "Alpaca 15-minute equity at the regular-session close"
    equity = equity.reindex(sessions)
    if not np.isfinite(equity).all() or not equity.gt(0).all():
        raise ValueError("Alpaca equity is missing or invalid for a trading session")
    return equity, source


def ideal_portfolio(prices):
    """Hold the next period's 15 best realized returns at equal starting weights."""
    marks = prices.groupby(prices.index.to_period("W-FRI")).tail(1).index
    nav = pd.Series(index=prices.index, dtype=float)
    nav.loc[marks[0]] = 1
    picks, weeks = [], []
    for start, end in pairwise(marks):
        period = prices.loc[start:end]
        usable = (np.isfinite(period) & period.gt(0)).all(axis=0)
        relative = period.loc[:, usable].div(period.loc[start, usable])
        returns = relative.loc[end].sub(1).sort_index()
        winners = returns.sort_values(ascending=False, kind="stable").head(15)
        if len(winners) != 15:
            raise ValueError(f"Insufficient eligible prices for {end.date()}")
        nav.loc[start:end] = nav.loc[start] * relative[winners.index].mean(axis=1)
        weeks.append(
            {
                "start": start,
                "end": end,
                "candidates": int(usable.sum()),
                "ideal_return": winners.mean(),
            }
        )
        picks.extend(
            {
                "start": start,
                "end": end,
                "rank": rank,
                "symbol": symbol,
                "return": value,
            }
            for rank, (symbol, value) in enumerate(winners.items(), 1)
        )
    if nav.isna().any():
        raise ValueError("Hindsight NAV is missing a trading day")
    return nav, pd.DataFrame(weeks), pd.DataFrame(picks)


def plot(growth):
    returns = growth - 1
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 7), layout="constrained")
    for ax, names, title in zip(
        axes,
        (("v1", "spy", "ideal"), ("v1", "spy")),
        ("All three portfolios", "V1 and SPY on a closer scale"),
        strict=True,
    ):
        for name in names:
            ax.plot(
                returns.index,
                returns[name],
                label=LABELS[name],
                color=COLORS[name],
                linewidth=2,
            )
        ax.axhline(0, color="#888888", linewidth=0.8)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_ylabel("Cumulative return")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="upper left", frameon=False)
    locator = MonthLocator()
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(ConciseDateFormatter(locator))
    fig.suptitle(
        f"US Alpha-HRP since the close-only switch · {growth.index[0]:%b %d}–{growth.index[-1]:%b %d, %Y}"
    )
    fig.savefig(OUTPUT / "comparison.png", dpi=165)
    plt.close(fig)


def main():
    evidence = json.loads((OUTPUT / "alpaca_evidence.json").read_text())
    end = pd.Timestamp(evidence["end"])
    calendar = xcals.get_calendar("XNYS")
    sessions = calendar.sessions_in_range(START, end)
    equity, terminal_source = broker_equity(evidence, sessions, calendar)
    closes = pd.read_csv(OUTPUT / "adjusted_close.csv", parse_dates=["date"]).set_index(
        "date"
    )
    if not closes.index.equals(sessions):
        raise ValueError("Price sessions do not match the US exchange calendar")
    spy = closes.pop("SPY")
    if not np.isfinite(spy).all() or not spy.gt(0).all():
        raise ValueError("SPY is missing an adjusted close")
    ideal, weeks, picks = ideal_portfolio(closes)
    activities = pd.DataFrame(evidence["activities"])
    if activities.id.duplicated().any():
        raise ValueError("Duplicate Alpaca activity IDs")
    funding = activities.activity_type.isin(
        ("CSD", "CSW", "TRANS", "JNLC", "JNLS", "ACATC", "ACATS")
    )
    if funding.any():
        raise ValueError("External funding or transfers require a cash-flow adjustment")
    fills = activities.loc[activities.activity_type.eq("FILL")].copy()
    fills["notional"] = fills.qty.astype(float) * fills.price.astype(float)
    fills["session"] = (
        pd.to_datetime(fills.transaction_time, format="ISO8601", utc=True)
        .dt.tz_convert(NY)
        .dt.tz_localize(None)
        .dt.normalize()
    )
    fills[
        [
            "session",
            "transaction_time",
            "order_id",
            "symbol",
            "side",
            "type",
            "qty",
            "price",
            "notional",
        ]
    ].to_csv(OUTPUT / "fills.csv", index=False)
    growth = pd.DataFrame(
        {"v1": equity / equity.iloc[0], "spy": spy / spy.iloc[0], "ideal": ideal}
    )
    growth.rename_axis("date").assign(alpaca_equity=equity).to_csv(
        OUTPUT / "daily_comparison.csv"
    )
    for name in ("v1", "spy"):
        weeks[f"{name}_return"] = (
            growth.loc[weeks.end, name].to_numpy()
            / growth.loc[weeks.start, name].to_numpy()
            - 1
        )
    weeks.to_csv(OUTPUT / "weekly_comparison.csv", index=False)
    picks.to_csv(OUTPUT / "hindsight_picks.csv", index=False)
    plot(growth)
    write_json(
        OUTPUT / "summary.json",
        {
            "switch_commit": "4e91cd1",
            "switch_date": "2026-08-23",
            "first_close_only_trade_session": str(fills.session.min().date()),
            "start": str(START.date()),
            "end": str(end.date()),
            "terminal_v1_source": terminal_source,
            "daily_marks": len(growth),
            "weeks": len(weeks),
            "candidate_range": [
                int(weeks.candidates.min()),
                int(weeks.candidates.max()),
            ],
            "initial_equity": equity.iloc[0],
            "ending_equity": equity.iloc[-1],
            "cumulative_return": (growth.iloc[-1] - 1).to_dict(),
            "max_drawdown": (growth.div(growth.cummax()) - 1).min().to_dict(),
            "fill_activities": fills.side.value_counts().to_dict(),
            "unique_filled_orders": fills.order_id.nunique(),
            "gross_fill_notional": fills.notional.sum(),
            "activity_types": activities.activity_type.value_counts().to_dict(),
        },
    )


if __name__ == "__main__":
    main()

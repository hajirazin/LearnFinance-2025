"""Compare actual Alpaca HRP performance against SPY around the close-only switch."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import yfinance as yf
from build_friday_dataset import plt
from compare_patchtst import write_json
from fetch_v1_production_comparison import NY, OUTPUT, START, fetch_broker
from matplotlib.ticker import PercentFormatter
from plot_v1_production_comparison import broker_equity

JANUARY = pd.Timestamp("2026-01-02")
ALPHA_BASELINE = pd.Timestamp("2026-05-01")
EVIDENCE = OUTPUT / "alpaca_jan_nov_2026.json"
SPY_PRICES = OUTPUT / "spy_jan_nov_2026.csv"


def fetch_spy(end):
    yf.set_tz_cache_location(str(OUTPUT / "yfinance_cache"))
    bars = yf.download(
        "SPY",
        start=JANUARY.date().isoformat(),
        end=(end.date() + timedelta(days=1)).isoformat(),
        auto_adjust=False,
        progress=False,
    )
    bars["Adj Close"]["SPY"].rename("adjusted_close").rename_axis("date").to_csv(
        SPY_PRICES
    )


def summarize(daily, weekly, start, end):
    window = weekly.loc[
        weekly.index.to_series().gt(start) & weekly.index.to_series().le(end)
    ]
    down = window.spy.lt(0)
    ahead = window.excess.gt(0)
    returns = daily.loc[end] / daily.loc[start] - 1
    return {
        "start": start.date().isoformat(),
        "end": end.date().isoformat(),
        "account_return": returns.alpaca,
        "spy_return": returns.spy,
        "cumulative_excess": returns.alpaca - returns.spy,
        "relative_wealth_return": (1 + returns.alpaca) / (1 + returns.spy) - 1,
        "weekly_mean_excess": window.excess.mean(),
        "weekly_median_excess": window.excess.median(),
        "weeks": len(window),
        "ahead": int(ahead.sum()),
        "spy_down_weeks": int(down.sum()),
        "ahead_on_spy_down": int((ahead & down).sum()),
        "mean_excess_on_spy_down": window.loc[down, "excess"].mean(),
    }


def chart(daily, weekly):
    fig, (top, bottom) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 6.5), layout="constrained"
    )
    relative = daily.alpaca / daily.spy
    top.plot(
        relative.index, relative / relative.iloc[0] - 1, color="#315c88", linewidth=2
    )
    top.set_title(
        "Alpaca equity relative to SPY · rebased at the first trading week", loc="left"
    )
    top.set_ylabel("Relative wealth return")
    bottom.bar(
        weekly.index,
        weekly.excess,
        width=4,
        color=np.where(weekly.excess.gt(0), "#4a7754", "#ae5a43"),
    )
    bottom.set_title("Weekly Alpaca return minus SPY return", loc="left")
    bottom.set_ylabel("Excess return")
    for ax in (top, bottom):
        ax.axhline(0, color="#808080", linewidth=0.8)
        ax.axvline(ALPHA_BASELINE, color="#737373", linestyle=":", linewidth=1.2)
        ax.axvline(START, color="#242f42", linestyle="--", linewidth=1.2)
        ax.grid(axis="y", alpha=0.2)
    top.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    bottom.yaxis.set_major_formatter(PercentFormatter(1, decimals=0, symbol=" pp"))
    for day, label in (
        (ALPHA_BASELINE, "Alpha-HRP starts"),
        (START, "Close-only starts"),
    ):
        top.annotate(
            label,
            (day, 0.97),
            xycoords=("data", "axes fraction"),
            xytext=(5, 0),
            textcoords="offset points",
            va="top",
            fontsize=9,
        )
    fig.suptitle(f"US HRP paper account vs. SPY · January–{daily.index[-1]:%B %Y}")
    fig.savefig(OUTPUT / "before_after_spy.png", dpi=165)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-through", type=pd.Timestamp, metavar="YYYY-MM-DD")
    args = parser.parse_args()
    if args.refresh_through is not None:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = (
                pool.submit(
                    fetch_broker, args.refresh_through, start=JANUARY, output=EVIDENCE
                ),
                pool.submit(fetch_spy, args.refresh_through),
            )
            for job in jobs:
                job.result()
    evidence = json.loads(EVIDENCE.read_text())
    activities = pd.DataFrame(evidence["activities"])
    if activities.id.duplicated().any():
        raise ValueError("Duplicate Alpaca activity IDs")
    fills = activities.loc[activities.activity_type.eq("FILL"), "transaction_time"]
    first_fill = pd.to_datetime(fills, format="ISO8601", utc=True).min().tz_convert(NY)
    calendar = xcals.get_calendar("XNYS")
    first_trade = pd.Timestamp(first_fill.date())
    start = calendar.previous_session(first_trade)
    funding = activities.loc[
        activities.activity_type.isin(
            ("CSD", "CSW", "TRANS", "JNLC", "JNLS", "ACATC", "ACATS")
        )
    ]
    # Alpaca's single initial journal is already reflected in the pre-trade
    # equity mark. Any additional cash flow needs explicit treatment.
    if len(funding) != 1 or pd.Timestamp(funding.date.iloc[0]) != first_trade:
        raise ValueError("Funding needs reconciliation against the initial equity")
    history = evidence["history"]
    dates = pd.to_datetime(history["timestamp"], unit="s", utc=True).tz_convert(NY)
    before_trading = pd.Series(history["equity"], index=dates.date).loc[: start.date()]
    if not np.isclose(
        before_trading.iloc[-1], float(funding.net_amount.iloc[0]), atol=0.01, rtol=0
    ):
        raise ValueError(
            "The starting equity does not reconcile to the initial funding"
        )
    end = pd.Timestamp(evidence["end"])
    sessions = calendar.sessions_in_range(start, end)
    equity, terminal_source = broker_equity(evidence, sessions, calendar)
    prices = (
        pd.read_csv(SPY_PRICES, parse_dates=["date"]).set_index("date").adjusted_close
    )
    daily = pd.DataFrame({"alpaca": equity, "spy": prices.reindex(sessions)})
    if not np.isfinite(daily).all().all() or not daily.gt(0).all().all():
        raise ValueError("Incomplete daily comparison marks")
    marks = daily.groupby(daily.index.to_period("W-FRI")).tail(1)
    weekly = marks.pct_change().iloc[1:]
    weekly["excess"] = weekly.alpaca - weekly.spy
    weekly["period"] = np.select(
        [weekly.index <= ALPHA_BASELINE, weekly.index <= START],
        ["earlier_hrp", "alpha_hrp_before_switch"],
        default="alpha_hrp_close_only",
    )
    weekly.rename_axis("week_ending").to_csv(OUTPUT / "before_after_weekly.csv")
    summary = {
        "first_funded_equity_date": str(
            before_trading.loc[before_trading.gt(0)].index[0]
        ),
        "first_fill_date": first_trade.date().isoformat(),
        "terminal_v1_source": terminal_source,
        "periods": {
            "whole_account_before": summarize(daily, weekly, start, START),
            "alpha_hrp_before": summarize(daily, weekly, ALPHA_BASELINE, START),
            "alpha_hrp_after": summarize(daily, weekly, START, end),
        },
    }
    write_json(OUTPUT / "before_after_summary.json", summary)
    chart(daily, weekly)


if __name__ == "__main__":
    main()

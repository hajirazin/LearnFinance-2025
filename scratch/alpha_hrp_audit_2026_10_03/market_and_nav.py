"""Offline actual broker NAV versus downloaded public market evidence."""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent
broker = json.loads((OUT / "broker_history.json").read_text())
market = pickle.loads((OUT / "market_prices.pkl").read_bytes())
closes = pd.DataFrame({s: frame.close for s, frame in market.items()})
closes.index.name = "date"
closes.to_csv(OUT / "market_adjusted_closes.csv")
pd.concat(market, names=["symbol", "date"]).to_csv(OUT / "market_ohlcv.csv")
nav = {}
for name, account in broker["accounts"].items():
    history = account["portfolio_history"]
    index = (
        pd.to_datetime(history["timestamp"], unit="s", utc=True)
        .tz_convert("America/New_York")
        .tz_localize(None)
        .normalize()
    )
    nav[name] = pd.Series(history["equity"], index=index).replace(0, np.nan)
nav = pd.DataFrame(nav)
nav.index.name = "date"
nav.to_csv(OUT / "broker_daily_nav.csv")

start, end = pd.Timestamp("2026-08-21"), pd.Timestamp("2026-10-02")


def ret(series, first=start, last=end):
    return float((series.loc[last] / series.loc[first] - 1) * 100)


all_series = {
    **{s: nav[s] for s in nav},
    **{s: closes[s] for s in ["SPY", "QQQ", "SPUS", "HLAL"]},
}
weeks = pd.to_datetime(
    [
        "2026-08-21",
        "2026-08-28",
        "2026-09-04",
        "2026-09-11",
        "2026-09-18",
        "2026-09-25",
        "2026-10-02",
    ]
)
weekly = pd.DataFrame(
    {
        s: series.reindex(weeks).pct_change(fill_method=None) * 100
        for s, series in all_series.items()
    }
).iloc[1:]
weekly.index.name = "week_ending"
weekly.to_csv(OUT / "weekly_actual_returns.csv")
conditions = {}
for s in closes:
    series = closes[s].dropna()
    if start not in series or end not in series:
        continue
    sliced = series.loc[start:end]
    conditions[s] = {
        "start_level": float(series.loc[start]),
        "end_level": float(series.loc[end]),
        "change_pct": ret(series),
        "max_drawdown_pct": float((sliced / sliced.cummax() - 1).min() * 100),
        "annualized_20_session_log_vol_start_pct": float(
            np.log(series).diff().loc[:start].tail(20).std(ddof=1) * np.sqrt(252) * 100
        ),
        "annualized_20_session_log_vol_end_pct": float(
            np.log(series).diff().loc[:end].tail(20).std(ddof=1) * np.sqrt(252) * 100
        ),
    }
pd.DataFrame(conditions).T.to_csv(OUT / "market_conditions.csv")
periods = {}
for first in ["2026-01-16", "2026-07-01", "2026-08-14", "2026-08-21"]:
    last = "2026-08-21" if first != "2026-08-21" else "2026-10-02"
    periods[f"{first} through {last}"] = {
        s: ret(series, pd.Timestamp(first), pd.Timestamp(last))
        for s, series in all_series.items()
    }
result = {
    "start": str(start.date()),
    "end": str(end.date()),
    "actual_nav_returns_pct": {s: ret(nav[s]) for s in nav},
    "benchmark_returns_pct": {
        s: ret(closes[s]) for s in ["SPY", "QQQ", "SPUS", "HLAL"]
    },
    "weeks": weekly.reset_index()
    .assign(week_ending=lambda x: x.week_ending.dt.strftime("%Y-%m-%d"))
    .to_dict("records"),
    "market_conditions": conditions,
    "earlier_periods": periods,
    "caveats": [
        "NAV includes actual paper fills and broker marks. No CSD/CSW cash-flow entries returned.",
        "CTVA/Vylor corporate-action credit appears absent from dhrp, so its economic return is unresolved.",
        "Earlier hrp account history may cover older HRP strategy variants.",
        "Market prices are retrospective Yahoo auto-adjusted evidence, not original run snapshots.",
        "VIX/TNX levels are not investable total returns.",
    ],
}
(OUT / "market_and_nav.json").write_text(json.dumps(result, indent=2))
print(
    json.dumps(
        {
            "returns": result["actual_nav_returns_pct"],
            "benchmarks": result["benchmark_returns_pct"],
            "earlier": periods,
        },
        indent=2,
    )
)
print(weekly.round(3).to_string())
print(
    pd.DataFrame(conditions)
    .T[["change_pct", "start_level", "end_level"]]
    .round(3)
    .to_string()
)

"""Fetch immutable experiment price evidence; never touch production stores."""

import json

import yfinance as yf
from experiment_settings import BENCHMARK, ROOT, UNIVERSE

REPO = ROOT.parents[1]


def main():
    roster = json.loads(
        (REPO / f"brain_api/data/cache/universe/{UNIVERSE}_2026-11.json").read_text()
    )
    symbols = sorted(stock["symbol"] for stock in roster["stocks"])
    destination = ROOT / "evidence"
    destination.mkdir(exist_ok=True)
    yf.set_tz_cache_location(str(destination / "yfinance_cache"))
    requested = sorted({*symbols, "SPY", "^VIX", BENCHMARK})
    prices = yf.download(
        requested,
        start="2016-01-01",
        end="2026-11-28",
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=False,
        timeout=30,
    )
    if prices.empty:
        raise ValueError("Yahoo returned no experiment price evidence")
    prices.to_parquet(destination / "adjusted_ohlcv.parquet")
    (destination / "roster.json").write_text(json.dumps(roster, indent=2))
    print("Saved adjusted OHLCV", prices.shape, prices.index.min(), prices.index.max())
    for symbol in requested:
        frame = prices[symbol].dropna(subset=["Open", "Close"])
        print(symbol, len(frame), str(frame.index.min()), str(frame.index.max()))


if __name__ == "__main__":
    main()

"""Fetch a separate 2026 snapshot for the V1 comparison."""

import argparse
import logging
import os
from datetime import date, datetime, time

from build_dataset import DATA, REPO, ROSTER, load_roster

OUTPUT = DATA / "v1_holdout"
PRICE_START = date(2025, 12, 15)
PRICE_END = date(2026, 11, 10)
NEWS_START = date(2026, 7, 6)
NEWS_END = date(2026, 11, 2)


def fetch_prices(symbols):
    import pandas as pd
    import yfinance as yf
    from yfinance.utils import auto_adjust

    yf.set_tz_cache_location(str(OUTPUT / "yfinance_cache"))
    path = OUTPUT / "prices_2026.csv"
    parts = []
    for i in range(0, len(symbols), 40):
        batch = symbols[i : i + 40]
        bars = yf.download(
            batch,
            start=PRICE_START.isoformat(),
            end=PRICE_END.isoformat(),
            auto_adjust=False,
            progress=False,
            threads=False,
            group_by="ticker",
        )
        for symbol in batch:
            if symbol not in bars.columns.get_level_values(0):
                continue
            raw = bars[symbol].dropna(
                subset=["Open", "High", "Low", "Close", "Adj Close", "Volume"]
            )
            if raw.empty:
                continue
            adjusted = auto_adjust(raw.copy()).rename(columns=str.lower)
            adjusted["raw_open"] = raw["Open"]
            adjusted["symbol"] = symbol
            parts.append(adjusted.rename_axis("date").reset_index())
        print(
            f"Prices requested for {i + len(batch)}/{len(symbols)} stocks", flush=True
        )
    frame = pd.concat(parts, ignore_index=True)
    if frame.duplicated(["date", "symbol"]).any():
        raise ValueError("Duplicate daily bars")
    last = frame.loc[frame["date"].eq(pd.Timestamp("2026-11-09")), "symbol"].nunique()
    if last < 300:
        raise ValueError(f"Only {last} stocks have a terminal-day price")
    temp = path.with_suffix(".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)
    print(
        f"Saved {len(frame):,} bars for {frame['symbol'].nunique()} stocks", flush=True
    )


def fetch_news(symbols, start_date=NEWS_START, end_date=NEWS_END):
    from brain_api.core.config import get_alpaca_api_key, get_alpaca_api_secret
    from brain_api.core.weekly_decision import DECISION_TIMEZONE
    from brain_api.news.store import NewsStore
    from brain_api.news_etl.backfill import run_backfill
    from dotenv import load_dotenv

    load_dotenv(REPO / "brain_api/.env", override=False)
    if not (get_alpaca_api_key() and get_alpaca_api_secret()):
        raise SystemExit("The Alpaca historical-data credentials are missing")
    os.environ["HF_HUB_CACHE"] = str(DATA / "finbert_cache")
    os.environ["HF_HUB_OFFLINE"] = "1"
    start = datetime.combine(start_date, time(9), DECISION_TIMEZONE)
    end = datetime.combine(end_date, time(9), DECISION_TIMEZONE)
    run_backfill(symbols=symbols, start=start, end=end, store=NewsStore(OUTPUT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("prices", "news"))
    parser.add_argument("--news-start", type=date.fromisoformat, default=NEWS_START)
    parser.add_argument("--news-end", type=date.fromisoformat, default=NEWS_END)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    symbols = load_roster(ROSTER)["symbols"]
    logging.basicConfig(level=logging.INFO)
    if args.dataset == "prices":
        fetch_prices(symbols)
    else:
        fetch_news(symbols, args.news_start, args.news_end)


if __name__ == "__main__":
    main()

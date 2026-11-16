"""Save an adjusted-close and volume snapshot for the V1 lookback shadow run."""

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import exchange_calendars as xcals
import pandas as pd
import yfinance as yf
from build_dataset import ROSTER, load_roster
from compare_patchtst import sha256, write_json

RESULT = Path(__file__).resolve().parent / "data/v1_lookback_shadow"


def download(symbols, start, through, *, threads):
    bars = yf.download(
        symbols,
        start=start.date().isoformat(),
        end=(through + timedelta(days=1)).date().isoformat(),
        auto_adjust=False,
        progress=False,
        threads=threads,
        group_by="column",
    )
    if bars.empty:
        return pd.DataFrame(columns=["date", "symbol", "close", "volume"])
    bars.index = bars.index.tz_localize(None)
    columns = {"Adj Close": "close", "Volume": "volume"}
    return (
        pd.concat(
            {
                name: bars[field].rename_axis(index="date", columns="symbol").stack()
                for field, name in columns.items()
            },
            axis=1,
        )
        .dropna()
        .reset_index()
    )


def fetch(start, through):
    calendar = xcals.get_calendar("XNYS")
    if start >= through or not calendar.is_session(through):
        raise ValueError("Supply a valid date range ending on a trade date")
    if calendar.session_close(through) > pd.Timestamp.now(tz="UTC"):
        raise ValueError(f"The US session on {through.date()} has not closed")
    save_dir = RESULT / "prices"
    save_dir.mkdir(parents=True, exist_ok=True)
    output = save_dir / f"{start.date()}_through_{through.date()}.parquet"
    if output.exists():
        raise FileExistsError(output)
    yf.set_tz_cache_location(str(save_dir / "yfinance_cache"))
    roster = load_roster(ROSTER)
    symbols = sorted(set(roster["symbols"]) | {"SPY"})
    frames = []
    for offset in range(0, len(symbols), 40):
        batch = symbols[offset : offset + 40]
        tidy = download(batch, start, through, threads=6)
        if not tidy.empty:
            frames.append(tidy)
        print(f"Requested {offset + len(batch)}/{len(symbols)} symbols", flush=True)
    if not frames:
        raise ValueError("No historical prices returned")
    prices = pd.concat(frames, ignore_index=True)
    missing = set(symbols) - set(prices.loc[prices.date.eq(through), "symbol"])
    for symbol in sorted(missing):
        retry = download([symbol], start, through, threads=False)
        if retry.date.eq(through).any():
            prices = pd.concat(
                [prices.loc[prices.symbol.ne(symbol)], retry], ignore_index=True
            )
    prices = prices.sort_values(["date", "symbol"])
    last = prices.loc[prices.date.eq(through)]
    if "SPY" not in set(last.symbol) or len(last) < 300:
        raise ValueError(
            f"Insufficient snapshot coverage on {through.date()}: {len(last)} symbols"
        )
    tmp = output.with_suffix(".tmp.parquet")
    with duckdb.connect() as db:
        db.from_df(prices).write_parquet(str(tmp))
    tmp.replace(output)
    write_json(
        output.with_suffix(".json"),
        {
            "provider": "Yahoo Finance adjusted closes and reported share volume",
            "retrieved_at_utc": datetime.now(UTC).isoformat(),
            "roster_sha256": roster["snapshot_sha256"],
            "requested_symbols": len(symbols),
            "symbols_with_last_bar": len(last),
            "missing_last_bar": sorted(set(symbols) - set(last.symbol)),
            "start": start.date().isoformat(),
            "through": through.date().isoformat(),
            "sha256": sha256(output),
        },
    )
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=pd.Timestamp, required=True)
    parser.add_argument("--through", type=pd.Timestamp, required=True)
    args = parser.parse_args()
    fetch(args.start, args.through)

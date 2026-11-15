"""Build the frozen 2016 through 2025 US price and news datasets in scratch."""

import argparse
import fcntl
import hashlib
import json
import logging
import os
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

EXPERIMENT = Path(__file__).resolve().parent
REPO = EXPERIMENT.parents[1]
DATA = EXPERIMENT / "data"
ROSTER = REPO / "scratch/patchtst_full_universe_audit_2026_08_23/results/manifest.json"
PRICE_ARCHIVE = (
    REPO / "scratch/AlphaAllocationV1/alpha_hrp_audit_2026_10_03/all_adjusted_ohlcv.csv"
)
CASH_ENTRIES = {("JPY", "JAPANESE YEN"), ("HKD", "HONG KONG DOLLAR")}
START = date(2016, 1, 1)
END = date(2025, 12, 31)
PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]


def load_roster(path):
    raw = path.read_bytes()
    payload = json.loads(raw)
    snapshot = payload.get("universe", payload)
    symbols = set()
    excluded = []
    remapped = {}
    original = [row["symbol"] for row in snapshot["stocks"]]
    if not original or not all(original) or len(original) != len(set(original)):
        raise ValueError("Expected a nonempty universe of unique symbols")
    for row in snapshot["stocks"]:
        symbol, name = row["symbol"], row.get("name", "")
        if (symbol, name) in CASH_ENTRIES:
            excluded.append(symbol)
        elif (symbol, name) == ("CCO", "Cameco Corp"):
            remapped[symbol] = "CCJ"
            symbols.add("CCJ")
        else:
            symbols.add(symbol)
    return {
        "symbols": sorted(symbols),
        "original_symbol_count": len(original),
        "excluded_cash_symbols": sorted(excluded),
        "us_listing_remaps": remapped,
        "snapshot": os.path.relpath(path.resolve(), REPO),
        "snapshot_sha256": hashlib.sha256(raw).hexdigest(),
        "fetched_at_utc": snapshot.get("fetched_at_utc"),
    }


def build_prices(roster, *, recover_kvue):
    import pandas as pd
    import yfinance as yf
    from brain_api.core.prices import load_prices_yfinance

    DATA.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(DATA / "yfinance_cache"))
    symbols = roster["symbols"]
    coverage = {}
    with TemporaryDirectory(dir=DATA) as temp:
        csv_path = Path(temp) / "daily_ohlcv_2016_2025.csv"
        with csv_path.open("w") as output:
            output.write(",".join(["symbol", "date", *PRICE_COLUMNS]) + "\n")
            for offset in range(0, len(symbols), 40):
                batch = symbols[offset : offset + 40]
                frames = load_prices_yfinance(batch, START, END, log_prefix="[Scratch]")
                recovered_kvue = (
                    recover_kvue and "KVUE" in batch and "KVUE" not in frames
                )
                if recovered_kvue:
                    archived = pd.read_csv(PRICE_ARCHIVE, parse_dates=["date"])
                    frame = archived.loc[archived["symbol"].eq("KVUE")].set_index(
                        "date"
                    )
                    frame = frame.loc[str(START) : str(END)].dropna(
                        subset=PRICE_COLUMNS
                    )
                    if frame.empty:
                        raise RuntimeError(
                            "No Kenvue rows found in the requested price archive"
                        )
                    frames["KVUE"] = frame
                    print(
                        "Recovered Kenvue's 2025 adjusted OHLCV from the prior Yahoo audit"
                    )
                for symbol, frame in sorted(frames.items()):
                    days = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
                    if days.has_duplicates or not all(
                        START <= d <= END for d in days.date
                    ):
                        raise ValueError(f"Unexpected daily index for {symbol}")
                    rows = frame[PRICE_COLUMNS].reset_index(drop=True)
                    rows.insert(0, "date", days.date)
                    rows.insert(0, "symbol", symbol)
                    rows.sort_values("date").to_csv(output, index=False, header=False)
                    coverage[symbol] = {
                        "rows": len(rows),
                        "first_session": days.min().date().isoformat(),
                        "last_session": days.max().date().isoformat(),
                    }
                    if recovered_kvue and symbol == "KVUE":
                        coverage[symbol]["archived_source"] = str(
                            PRICE_ARCHIVE.relative_to(REPO)
                        )
        if not coverage:
            raise RuntimeError("Yahoo returned no OHLCV. No CSV was published.")
        manifest = {
            "universe": roster,
            "provider": "Yahoo Finance",
            "yfinance_version": yf.__version__,
            "interval": "1d",
            "auto_adjust": True,
            "start_inclusive": START.isoformat(),
            "end_inclusive": END.isoformat(),
            "downloaded_at_utc": datetime.now(UTC).isoformat(),
            "rows": sum(entry["rows"] for entry in coverage.values()),
            "missing_symbols": sorted(set(symbols) - coverage.keys()),
            "symbols_with_data": coverage,
        }
        with csv_path.open("rb") as output:
            manifest["csv_sha256"] = hashlib.file_digest(output, "sha256").hexdigest()
        manifest_path = Path(temp) / "ohlcv_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        csv_path.replace(DATA / csv_path.name)
        manifest_path.replace(DATA / manifest_path.name)
    print(
        f"Wrote {manifest['rows']} daily bars for {len(coverage)}/{len(symbols)} symbols"
    )
    if manifest["missing_symbols"]:
        print(f"Symbols without bars: {', '.join(manifest['missing_symbols'])}")


def export_weekly_news(store, symbols, windows):
    import duckdb
    import pandas as pd
    from brain_api.core.ppo_discovery.news_adapter import build_ppo_news_features

    store.require_coverage_many(symbols, windows)
    rows = []
    for offset in range(0, len(windows), 13):
        batch = windows[offset : offset + 13]
        for window, events in store.query_events_many(symbols, batch).items():
            by_symbol = {symbol: [] for symbol in symbols}
            for event in events:
                by_symbol[event.symbol].append(event)
            features = build_ppo_news_features(by_symbol, cutoff=window.end_inclusive)
            rows.extend(
                (
                    symbol,
                    window.start_exclusive,
                    window.end_inclusive,
                    f.raw_sentiment,
                    f.article_count,
                    f.log1p_article_count,
                    f.recency,
                    f.sentiment_dispersion,
                )
                for symbol, f in features.items()
            )
    weekly = pd.DataFrame.from_records(
        rows,
        columns=[
            "symbol",
            "window_start_exclusive",
            "window_end_inclusive",
            "raw_sentiment",
            "article_count",
            "log1p_article_count",
            "news_recency",
            "sentiment_dispersion",
        ],
    )
    if (
        len(weekly) != len(windows) * len(symbols)
        or weekly.duplicated(["symbol", "window_end_inclusive"]).any()
    ):
        raise RuntimeError(
            "Weekly news output does not match the expected symbol-weeks"
        )
    with duckdb.connect(str(store.db_path)) as connection:
        connection.register("weekly", weekly)
        connection.execute(
            "CREATE OR REPLACE TABLE weekly_sentiment AS SELECT * FROM weekly"
        )
    return len(weekly)


def build_news(roster):
    from brain_api.core.config import get_alpaca_api_key, get_alpaca_api_secret
    from brain_api.core.weekly_decision import (
        DECISION_TIMEZONE,
        canonical_monday_windows_contained_in,
    )
    from brain_api.news.models import NewsWindow
    from brain_api.news.store import NewsStore
    from brain_api.news_etl.backfill import run_backfill
    from dotenv import load_dotenv

    load_dotenv(REPO / "brain_api/.env", override=False)
    if not (get_alpaca_api_key() and get_alpaca_api_secret()):
        raise SystemExit(
            "Set ALPACA_API_KEY and ALPACA_API_SECRET to backfill Alpaca news"
        )
    os.environ["HF_HUB_CACHE"] = str(DATA / "finbert_cache")
    symbols = roster["symbols"]
    start = datetime.combine(START, time.min, DECISION_TIMEZONE)
    end = datetime.combine(END + timedelta(days=1), time.min, DECISION_TIMEZONE)
    windows = [
        NewsWindow(a, b) for a, b in canonical_monday_windows_contained_in(start, end)
    ]
    store = NewsStore(DATA)
    job_id = run_backfill(symbols=symbols, start=start, end=end, store=store)
    rows = export_weekly_news(store, symbols, windows)
    print(f"Wrote {rows} weekly sentiment rows to {store.db_path} (job {job_id})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("prices", "news"))
    parser.add_argument("--roster", type=Path, default=ROSTER)
    parser.add_argument(
        "--recover-kvue-from-audit",
        action="store_true",
        help="Use the previous Yahoo audit's 2025 Kenvue bars if live Yahoo returns none",
    )
    args = parser.parse_args()
    if args.recover_kvue_from_audit and args.dataset != "prices":
        parser.error("--recover-kvue-from-audit applies to prices only")
    logging.basicConfig(level=logging.INFO)
    roster = load_roster(args.roster)
    DATA.mkdir(parents=True, exist_ok=True)
    with (DATA / f"{args.dataset}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(
                f"A scratch {args.dataset} build is already running"
            ) from None
        if args.dataset == "prices":
            build_prices(roster, recover_kvue=args.recover_kvue_from_audit)
        else:
            build_news(roster)


if __name__ == "__main__":
    main()

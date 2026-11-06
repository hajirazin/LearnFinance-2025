"""Public adjusted-price evidence for the 2025-cutoff research retrain."""

import json
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def main():
    bucket = REPO / "brain_api/data/models/ppo_discovery_halal_new"
    version = (bucket / "current").read_text().strip()
    roster = json.loads((bucket / version / "universe_manifest.json").read_text())
    symbols = [*roster["sorted_symbols"], "SPY", "^VIX"]
    yf.set_tz_cache_location(str(ROOT / "yfinance_cache"))
    raw = yf.download(
        symbols,
        start="2019-01-01",
        end="2026-09-15",
        auto_adjust=True,
        group_by="ticker",
        threads=8,
        progress=False,
        timeout=15,
    )
    if raw.empty:
        raise RuntimeError("Public historical download returned no prices")
    frames = {}
    for symbol in symbols:
        frame = raw[symbol].reindex(columns=["Open", "High", "Low", "Close", "Volume"])
        frame.columns = ["open", "high", "low", "close", "volume"]
        frames[symbol] = frame.dropna().copy()
    pd.to_pickle(frames, ROOT / "market_prices.pkl")
    (ROOT / "market_roster.json").write_text(json.dumps(roster, indent=2))
    (ROOT / "market_download.json").write_text(
        json.dumps(
            {
                "retrieved_at": pd.Timestamp.now(tz="UTC").isoformat(),
                "source": "yfinance auto_adjust=True",
                "source_roster_version": version,
                "start": "2019-01-01",
                "end": "2026-09-14",
                "rows": {s: len(f) for s, f in frames.items()},
                "empty_symbols": [s for s, f in frames.items() if f.empty],
            },
            indent=2,
        )
    )
    print("Downloaded", len(frames), "frames", flush=True)


if __name__ == "__main__":
    main()

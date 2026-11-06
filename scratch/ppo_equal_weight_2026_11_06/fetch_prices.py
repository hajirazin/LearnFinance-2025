"""Download public research evidence into scratch; never touch model storage."""

import json
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
BUCKET = REPO / "brain_api/data/models/ppo_discovery_halal_new"


def main():
    version = (BUCKET / "current").read_text().strip()
    artifact = BUCKET / version
    manifest = json.loads((artifact / "universe_manifest.json").read_text())
    metadata = json.loads((artifact / "metadata.json").read_text())
    symbols = [*manifest["sorted_symbols"], "SPY", "^VIX"]
    end = pd.Timestamp(metadata["end_date"]) + pd.Timedelta(days=1)
    yf.set_tz_cache_location(str(ROOT / "yfinance_cache"))
    data = yf.download(
        symbols,
        start="2019-01-01",
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=8,
        timeout=15,
    )
    if data.empty:
        raise RuntimeError("Public price download returned no evidence")
    prices = {}
    for symbol in symbols:
        frame = data[symbol].reindex(columns=["Open", "High", "Low", "Close", "Volume"])
        frame.columns = ["open", "high", "low", "close", "volume"]
        prices[symbol] = frame.dropna().copy()
    pd.to_pickle(prices, ROOT / "prices.pkl")
    summary = {
        s: {"rows": len(f), "first": str(f.index.min()), "last": str(f.index.max())}
        for s, f in prices.items()
    }
    (ROOT / "download_manifest.json").write_text(
        json.dumps(
            {
                "version": version,
                "source": "yfinance auto_adjust=True",
                "retrieved_at": pd.Timestamp.now(tz="UTC").isoformat(),
                "symbols": summary,
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                "loaded": sum(len(f) > 0 for f in prices.values()),
                "requested": len(symbols),
                "empty": [s for s, f in prices.items() if f.empty],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

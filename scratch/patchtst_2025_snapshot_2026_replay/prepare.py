"""Freeze Yahoo adjusted OHLCV under scratch; never write production evidence."""

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
DATA = REPO / "brain_api/data"
VERSION = "snapshot-2025-12-31-12d23440a2a3"
SNAPSHOT = DATA / "models/patchtst_halal_new" / VERSION
START = "2023-11-01"
END_EXCLUSIVE = "2026-11-28"


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def main():
    root = ROOT / "data/prices"
    root.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(ROOT / "data/yfinance_cache"))
    metadata = json.loads((SNAPSHOT / "metadata.json").read_text())
    symbols = list(dict.fromkeys([*metadata["symbols"], "SPY", "QQQ", "^GSPC"]))
    pending = [s for s in symbols if not (root / f"{s}.csv").exists()]
    errors = {}
    for offset in range(0, len(pending), 25):
        batch = pending[offset : offset + 25]
        print(
            f"Downloading batch {offset // 25 + 1}: {batch[0]} .. {batch[-1]}",
            flush=True,
        )
        frames = yf.download(
            batch,
            start=START,
            end=END_EXCLUSIVE,
            group_by="ticker",
            auto_adjust=True,
            threads=False,
            progress=False,
            timeout=30,
        )
        for symbol in batch:
            frame = (
                frames[symbol].copy()
                if isinstance(frames.columns, pd.MultiIndex)
                else frames.copy()
            )
            frame.columns = frame.columns.str.lower()
            frame = frame.reindex(
                columns=["open", "high", "low", "close", "volume"]
            ).dropna(how="all")
            if frame.empty:
                errors[symbol] = (
                    "Provider returned no OHLCV; not replaced with other evidence"
                )
                continue
            frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
            frame.to_csv(root / f"{symbol}.csv", index_label="date")
        time.sleep(0.25)
    manifest = {
        "retrieved_at": datetime.now(UTC).isoformat(),
        "provider": "Yahoo Finance via yfinance",
        "yfinance_version": yf.__version__,
        "auto_adjust": True,
        "start": START,
        "end_exclusive": END_EXCLUSIVE,
        "universe": metadata["symbols"],
        "snapshot": VERSION,
        "snapshot_sha256": {
            p.name: sha256(p) for p in SNAPSHOT.iterdir() if p.is_file()
        },
        "prices_sha256": {p.name: sha256(p) for p in root.glob("*.csv")},
        "provider_failures": errors,
    }
    write_json(ROOT / "price_manifest.json", manifest)
    print(
        f"Frozen {len(manifest['prices_sha256'])}/{len(symbols)} price histories; failures={errors}",
        flush=True,
    )


if __name__ == "__main__":
    main()

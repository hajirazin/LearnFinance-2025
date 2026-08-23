"""Auditable current-universe and adjusted-price acquisition for the sweep."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance
from pooling_spec import json_dump, sha256_file

from brain_api.core.prices import load_prices_yfinance

PriceDownloader = Callable[..., dict[str, pd.DataFrame]]


def load_halal_new_universe_cache(
    path: Path, *, minimum_symbols: int = 300
) -> tuple[list[str], dict[str, Any]]:
    """Load and fingerprint the existing production-built universe cache."""
    raw = json.loads(path.read_text())
    stocks = raw.get("stocks", [])
    symbols = [row["symbol"] for row in stocks]
    if len(symbols) != raw.get("total_stocks") or len(symbols) != len(set(symbols)):
        raise RuntimeError("halal_new cache count or uniqueness mismatch")
    if len(symbols) < minimum_symbols:
        raise RuntimeError(
            f"halal_new cache has only {len(symbols)} symbols; need {minimum_symbols}"
        )
    return symbols, {
        "source_kind": "existing_repository_cache",
        "source_path": str(path),
        "source_sha256": sha256_file(path),
        "fetched_at_utc": raw.get("fetched_at"),
        "source_etfs": raw.get("etfs_used", []),
        "halal_new_count": len(symbols),
        "stocks": stocks,
    }


def _safe_filename(symbol: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in symbol)


def _read_verified_cache(
    data_dir: Path,
    manifest_path: Path,
    symbols: list[str],
    start_date: date,
    end_date: date,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]] | None:
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "requested_symbols": symbols,
        "start_date": start_date.isoformat(),
        "end_date_exclusive": end_date.isoformat(),
        "auto_adjust": True,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        return None
    prices: dict[str, pd.DataFrame] = {}
    for symbol, info in manifest.get("files", {}).items():
        path = data_dir / info["file"]
        if not path.exists() or sha256_file(path) != info["sha256"]:
            raise RuntimeError(f"cached adjusted-price hash mismatch for {symbol}")
        prices[symbol] = pd.read_csv(path, index_col=0, parse_dates=True)
    return prices, manifest


def load_or_download_prices(
    symbols: list[str],
    *,
    data_dir: Path,
    start_date: date,
    end_date: date,
    chunk_size: int = 80,
    downloader: PriceDownloader = load_prices_yfinance,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Load a hash-verified cache or download every requested current symbol."""
    if len(symbols) != len(set(symbols)):
        raise ValueError("requested symbols contain duplicates")
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = data_dir / "adjusted_ohlcv_manifest.json"
    cached = _read_verified_cache(
        data_dir, manifest_path, symbols, start_date, end_date
    )
    if cached is not None:
        return cached

    downloaded: dict[str, pd.DataFrame] = {}
    for start in range(0, len(symbols), chunk_size):
        chunk = symbols[start : start + chunk_size]
        downloaded.update(
            downloader(
                chunk,
                start_date,
                end_date,
                log_prefix=f"[PatchPooling {start + 1}-{start + len(chunk)}]",
            )
        )

    files: dict[str, Any] = {}
    prices: dict[str, pd.DataFrame] = {}
    used_filenames: set[str] = set()
    for symbol, frame in sorted(downloaded.items()):
        required = ["open", "high", "low", "close", "volume"]
        if any(column not in frame.columns for column in required):
            raise ValueError(f"downloaded adjusted OHLCV columns missing for {symbol}")
        clean = frame[required].copy()
        index = pd.DatetimeIndex(pd.to_datetime(clean.index))
        if index.tz is not None:
            index = index.tz_localize(None)
        clean.index = index.normalize()
        clean = clean.sort_index()
        clean = clean[~clean.index.duplicated(keep="last")]
        filename = f"{_safe_filename(symbol)}_adjusted.csv"
        if filename in used_filenames:
            raise RuntimeError(f"cache filename collision for {symbol}")
        used_filenames.add(filename)
        path = data_dir / filename
        clean.to_csv(path, index_label="date", float_format="%.12g")
        normalized = pd.read_csv(path, index_col=0, parse_dates=True)
        prices[symbol] = normalized
        files[symbol] = {
            "file": filename,
            "sha256": sha256_file(path),
            "rows": len(normalized),
            "first_session": normalized.index.min().date().isoformat(),
            "last_session": normalized.index.max().date().isoformat(),
        }

    manifest = {
        "provider": "yfinance",
        "provider_version": yfinance.__version__,
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "requested_symbols": symbols,
        "downloaded_symbols": sorted(prices),
        "missing_symbols": [symbol for symbol in symbols if symbol not in prices],
        "start_date": start_date.isoformat(),
        "end_date_exclusive": end_date.isoformat(),
        "auto_adjust": True,
        "files": files,
    }
    json_dump(manifest_path, manifest)
    return prices, manifest

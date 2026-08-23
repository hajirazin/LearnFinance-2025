"""Auditable halal_new universe and adjusted OHLCV acquisition."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance
from dotenv import load_dotenv
from experiment_spec import json_dump, sha256_file

from brain_api.core.prices import load_prices_yfinance
from brain_api.universe.halal_new import (
    ALL_ETFS,
    HALAL_NEW_ETF_NAMES,
    SP_FUNDS_ETFS,
    WAHED_ETFS,
    _merge_and_dedup,
)
from brain_api.universe.scrapers import (
    fetch_alpaca_tradable_symbols,
    scrape_sp_funds,
    scrape_wahed,
)


def load_halal_new_universe_cache(
    path: Path, *, minimum_symbols: int = 300
) -> tuple[list[str], dict[str, Any]]:
    """Load and fingerprint an existing repository-produced halal_new cache."""
    raw = json.loads(path.read_text())
    stocks = raw.get("stocks", [])
    symbols = [row["symbol"] for row in stocks]
    if len(symbols) != raw.get("total_stocks") or len(symbols) != len(set(symbols)):
        raise RuntimeError("halal_new cache count or uniqueness mismatch")
    if len(symbols) < minimum_symbols:
        raise RuntimeError(
            f"halal_new cache has only {len(symbols)} symbols; need {minimum_symbols}"
        )
    manifest = {
        "source_kind": "existing_repository_cache",
        "source_path": str(path),
        "source_sha256": sha256_file(path),
        "fetched_at_utc": raw.get("fetched_at"),
        "source_etfs": raw.get("etfs_used", []),
        "halal_new_count": len(symbols),
        "stocks": stocks,
    }
    return symbols, manifest


def fetch_uncached_halal_new_universe(
    *,
    brain_env_path: Path,
) -> tuple[list[str], dict[str, Any]]:
    """Rebuild halal_new without writing the production universe cache."""
    load_dotenv(brain_env_path, override=False)
    holdings: dict[str, list[dict[str, Any]]] = {}
    for slug in SP_FUNDS_ETFS:
        holdings[slug.upper()] = scrape_sp_funds(slug)
    for slug in WAHED_ETFS:
        holdings[slug.upper()] = scrape_wahed(slug)
    merged = _merge_and_dedup(holdings)
    alpaca_symbols = fetch_alpaca_tradable_symbols()
    tradable = [row for row in merged if row["symbol"] in alpaca_symbols]
    existing = {row["symbol"] for row in tradable}
    for ticker in [slug.upper() for slug in ALL_ETFS]:
        if ticker not in existing and ticker in alpaca_symbols:
            tradable.append(
                {
                    "symbol": ticker,
                    "name": HALAL_NEW_ETF_NAMES[ticker],
                    "max_weight": 0.0,
                    "sources": ["etf-self"],
                }
            )
    symbols = [row["symbol"] for row in tradable]
    if len(symbols) != len(set(symbols)):
        raise RuntimeError("halal_new universe contains duplicate symbols")
    if len(symbols) < 300:
        raise RuntimeError(
            f"halal_new unexpectedly has only {len(symbols)} tradable symbols"
        )
    manifest = {
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "source_etfs": [slug.upper() for slug in ALL_ETFS],
        "holdings_counts": {name: len(rows) for name, rows in holdings.items()},
        "merged_count": len(merged),
        "alpaca_tradable_asset_count": len(alpaca_symbols),
        "halal_new_count": len(symbols),
        "stocks": tradable,
    }
    return symbols, manifest


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
            raise RuntimeError(f"cached OHLCV hash mismatch for {symbol}")
        prices[symbol] = pd.read_csv(path, index_col=0, parse_dates=True)
    return prices, manifest


def load_or_download_prices(
    symbols: list[str],
    *,
    data_dir: Path,
    start_date: date,
    end_date: date,
    chunk_size: int = 80,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Load a verified cache or download the complete requested symbol list."""
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
            load_prices_yfinance(
                chunk,
                start_date,
                end_date,
                log_prefix=f"[FullUniverse {start + 1}-{start + len(chunk)}]",
            )
        )

    files: dict[str, Any] = {}
    prices: dict[str, pd.DataFrame] = {}
    used_filenames: set[str] = set()
    for symbol, frame in sorted(downloaded.items()):
        clean = frame[["open", "high", "low", "close", "volume"]].copy()
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

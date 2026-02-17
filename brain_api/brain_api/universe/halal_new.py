"""Halal_New stock universe from full ETF holdings.

Scrapes all holdings from 5 halal ETFs (SPUS, SPTE, SPWO from SP Funds;
HLAL, UMMA from Wahed), merges and deduplicates, then filters to only
symbols tradable on Alpaca.

This produces a much larger universe (~410 stocks) compared to the
original halal universe (~45 stocks from yfinance top holdings).
"""

import logging
from datetime import UTC, datetime

from brain_api.universe.scrapers import (
    fetch_alpaca_tradable_symbols,
    scrape_sp_funds,
    scrape_wahed,
)

logger = logging.getLogger(__name__)

# ETF slugs grouped by data source
SP_FUNDS_ETFS = ["spus", "spte", "spwo"]
WAHED_ETFS = ["hlal", "umma"]
ALL_ETFS = SP_FUNDS_ETFS + WAHED_ETFS


def _merge_and_dedup(
    etf_holdings: dict[str, list[dict]],
) -> list[dict]:
    """Merge holdings from multiple ETFs, deduplicate by symbol, track sources.

    For duplicate symbols across ETFs, keeps the maximum weight and records
    all ETF sources.

    Args:
        etf_holdings: Mapping of ETF name -> list of holdings dicts.

    Returns:
        Deduplicated list sorted by max_weight descending.
    """
    merged: dict[str, dict] = {}

    for etf_name, holdings in etf_holdings.items():
        for h in holdings:
            symbol = h["symbol"]
            if symbol not in merged:
                merged[symbol] = {
                    "symbol": symbol,
                    "name": h["name"],
                    "max_weight": h["weight"],
                    "sources": [etf_name],
                }
            else:
                existing = merged[symbol]
                existing["max_weight"] = max(existing["max_weight"], h["weight"])
                if etf_name not in existing["sources"]:
                    existing["sources"].append(etf_name)

    return sorted(merged.values(), key=lambda x: x["max_weight"], reverse=True)


def get_halal_new_universe() -> dict:
    """Scrape 5 halal ETFs, merge, and filter to Alpaca-tradable.

    Pipeline:
        1. Scrape SPUS, SPTE, SPWO from sp-funds.com
        2. Fetch HLAL, UMMA from Wahed Google Sheets
        3. Merge and deduplicate across all ETFs
        4. Fetch Alpaca tradable symbols
        5. Keep only tradable symbols

    Returns:
        Dict with:
        - stocks: deduplicated list sorted by max_weight desc
        - etfs_used: list of ETF slugs scraped
        - total_stocks: count of tradable unique stocks
        - fetched_at: ISO timestamp
    """
    # 1 & 2: Scrape all ETFs
    etf_data: dict[str, list[dict]] = {}

    for slug in SP_FUNDS_ETFS:
        etf_data[slug.upper()] = scrape_sp_funds(slug)

    for slug in WAHED_ETFS:
        etf_data[slug.upper()] = scrape_wahed(slug)

    # 3: Merge and deduplicate
    merged = _merge_and_dedup(etf_data)
    logger.info(f"Merged {len(merged)} unique tickers from {len(ALL_ETFS)} ETFs")

    # 4: Fetch Alpaca tradable symbols
    alpaca_symbols = fetch_alpaca_tradable_symbols()

    # 5: Filter to tradable only
    tradable = [h for h in merged if h["symbol"] in alpaca_symbols]
    logger.info(
        f"Halal_New universe: {len(tradable)} tradable out of "
        f"{len(merged)} total scraped"
    )

    return {
        "stocks": tradable,
        "etfs_used": [s.upper() for s in ALL_ETFS],
        "total_stocks": len(tradable),
        "fetched_at": datetime.now(UTC).isoformat(),
    }


def get_halal_new_symbols() -> list[str]:
    """Get just the list of Halal_New stock symbols.

    Convenience function for use by training pipelines.

    Returns:
        List of tradable halal stock symbols from the expanded universe.
    """
    universe = get_halal_new_universe()
    return [s["symbol"] for s in universe["stocks"]]

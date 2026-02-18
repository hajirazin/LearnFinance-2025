"""Halal stock universe from ETF holdings.

Fetches top holdings from halal ETFs (SPUS, HLAL, SPTE) and filters to
US-listed stocks with Alpha Vantage fundamentals data available.
"""

import logging
from datetime import UTC, datetime

import yfinance as yf

from brain_api.universe.cache import load_cached_universe, save_universe_cache

logger = logging.getLogger(__name__)

# Halal ETFs to source holdings from
HALAL_ETFS = ["SPUS", "HLAL", "SPTE"]

# Tickers to exclude
# - GOOG: Alphabet Class C (duplicate of GOOGL Class A)
# - ATEYY: Advantest OTC ADR (no Alpha Vantage data)
EXCLUDED_TICKERS = {"GOOG", "ATEYY"}


def is_us_ticker(symbol: str) -> bool:
    """Check if a ticker appears to be US-listed.

    US tickers are typically 1-5 uppercase letters without:
    - Dots (e.g., ASML.AS is Amsterdam, 2454.TW is Taiwan)
    - Leading numbers (e.g., 2454.TW, 2308.TW)

    Args:
        symbol: Stock ticker symbol

    Returns:
        True if the ticker appears to be US-listed
    """
    if not symbol:
        return False
    if "." in symbol:
        return False
    return not symbol[0].isdigit()


def _fetch_etf_holdings(ticker: str) -> list[dict]:
    """Fetch top holdings for a single ETF using yfinance.

    Args:
        ticker: ETF ticker symbol

    Returns:
        List of holdings with symbol, name, weight
    """
    try:
        etf = yf.Ticker(ticker)
        if not hasattr(etf, "funds_data") or etf.funds_data is None:
            return []

        top_holdings = etf.funds_data.top_holdings
        if top_holdings is None or top_holdings.empty:
            return []

        rows = []
        for symbol, row in top_holdings.iterrows():
            # yfinance returns weight as decimal (0.15 = 15%)
            weight_raw = row.get("Holding Percent", row.get("holdingPercent", 0))
            weight = float(weight_raw) * 100 if weight_raw else 0

            rows.append(
                {
                    "symbol": str(symbol),
                    "name": row.get("Name", row.get("holdingName", "")),
                    "weight": weight,
                }
            )
        return rows
    except Exception:
        return []


def get_halal_universe() -> dict:
    """Fetch and merge top holdings from halal ETFs.

    Filters to US-listed stocks only and removes duplicate tickers
    (e.g., GOOG/GOOGL - keeps GOOGL).

    Returns:
        Dict with:
        - stocks: deduplicated list sorted by max_weight desc
        - etfs_used: list of ETF tickers queried
        - total_stocks: count of unique stocks
        - fetched_at: ISO timestamp
    """
    cached = load_cached_universe("halal")
    if cached is not None:
        return cached

    # Collect holdings from all ETFs
    all_holdings: dict[str, dict] = {}

    for etf_ticker in HALAL_ETFS:
        holdings = _fetch_etf_holdings(etf_ticker)
        logger.info(f"[{etf_ticker}] Raw holdings: {[h['symbol'] for h in holdings]}")
        for h in holdings:
            symbol = h["symbol"]

            # Skip non-US tickers and duplicates
            if not is_us_ticker(symbol):
                logger.debug(f"Filtered out non-US ticker: {symbol}")
                continue
            if symbol in EXCLUDED_TICKERS:
                logger.debug(f"Filtered out excluded ticker: {symbol}")
                continue

            if symbol not in all_holdings:
                all_holdings[symbol] = {
                    "symbol": symbol,
                    "name": h["name"],
                    "max_weight": h["weight"],
                    "sources": [etf_ticker],
                }
            else:
                # Update max weight and add source
                existing = all_holdings[symbol]
                existing["max_weight"] = max(existing["max_weight"], h["weight"])
                if etf_ticker not in existing["sources"]:
                    existing["sources"].append(etf_ticker)

    # Sort by max_weight descending
    stocks = sorted(all_holdings.values(), key=lambda x: x["max_weight"], reverse=True)
    logger.info(
        f"Final US universe: {len(stocks)} stocks: {[s['symbol'] for s in stocks]}"
    )

    result = {
        "stocks": stocks,
        "etfs_used": HALAL_ETFS,
        "total_stocks": len(stocks),
        "fetched_at": datetime.now(UTC).isoformat(),
    }
    save_universe_cache("halal", result)
    return result


def get_halal_symbols() -> list[str]:
    """Get just the list of halal stock symbols.

    Convenience function for use by ETL and other consumers.

    Returns:
        List of US halal stock symbols
    """
    universe = get_halal_universe()
    return [s["symbol"] for s in universe["stocks"]]

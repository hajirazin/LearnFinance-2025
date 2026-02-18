"""S&P 500 stock universe from datahub.io.

Fetches the current S&P 500 constituents from datahub.io's maintained dataset.
This provides a larger training universe (~500 stocks) for forecasting models.
"""

import logging
from datetime import UTC, datetime

import pandas as pd

from brain_api.universe.cache import load_cached_universe, save_universe_cache

logger = logging.getLogger(__name__)

# Datahub.io maintains an updated S&P 500 constituents CSV
SP500_CSV_URL = "https://datahub.io/core/s-and-p-500-companies-financials/_r/-/data/constituents.csv"


def get_sp500_universe() -> dict:
    """Fetch S&P 500 constituents from datahub.io.

    The datahub.io dataset is derived from Wikipedia's S&P 500 list
    and includes symbol, name, and sector for each company.

    Returns:
        Dict with:
        - stocks: list of stock dicts with symbol, name, sector
        - source: data source identifier
        - total_stocks: count of stocks
        - fetched_at: ISO timestamp

    Raises:
        Exception: If CSV fetch fails (network error, bad URL, etc.)
    """
    cached = load_cached_universe("sp500")
    if cached is not None:
        return cached

    logger.info(f"Fetching S&P 500 constituents from {SP500_CSV_URL}")

    df = pd.read_csv(SP500_CSV_URL)

    stocks = []
    for _, row in df.iterrows():
        symbol = str(row["Symbol"]).strip()
        # Some symbols have dots (e.g., BRK.B) - keep them as-is for S&P 500
        stocks.append(
            {
                "symbol": symbol,
                "name": str(row.get("Name", "")).strip(),
                "sector": str(row.get("Sector", "")).strip(),
            }
        )

    logger.info(f"Fetched {len(stocks)} S&P 500 constituents")

    result = {
        "stocks": stocks,
        "source": "datahub.io",
        "total_stocks": len(stocks),
        "fetched_at": datetime.now(UTC).isoformat(),
    }
    save_universe_cache("sp500", result)
    return result


def get_sp500_symbols() -> list[str]:
    """Get just the list of S&P 500 stock symbols.

    Convenience function for use by training and other consumers.

    Returns:
        List of S&P 500 stock symbols (~500 symbols)
    """
    universe = get_sp500_universe()
    return [s["symbol"] for s in universe["stocks"]]

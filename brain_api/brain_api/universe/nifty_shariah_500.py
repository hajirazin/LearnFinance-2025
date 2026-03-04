"""NiftyShariah500 stock universe: all Nifty 500 Shariah constituents.

Fetches the full Nifty 500 Shariah index from NSE India (~210 stocks)
and stores symbols with .NS suffix so every downstream consumer
(training, inference, HRP, email) gets yfinance-ready symbols
without any append/strip transformation.

This is the base universe for India, analogous to halal_new for the US.
halal_india selects the top 15 from this universe via PatchTST inference.
"""

import logging
import threading
from datetime import UTC, datetime

from brain_api.universe.cache import load_cached_universe, save_universe_cache
from brain_api.universe.scrapers.nse import scrape_nifty500_shariah

logger = logging.getLogger(__name__)

NS_SUFFIX = ".NS"
CACHE_NAME = "nifty_shariah_500"


def get_nifty_shariah_500_universe(
    shutdown_event: threading.Event | None = None,
) -> dict:
    """Fetch all Nifty 500 Shariah constituents (~210 stocks) from NSE India.

    Symbols include .NS suffix for yfinance compatibility. No filtering
    or scoring is applied -- this returns the raw index constituents.

    Args:
        shutdown_event: Reserved for future cancellation support.

    Returns:
        Dict with:
        - stocks: all constituents with .NS-suffixed symbols
        - source: "nifty_500_shariah"
        - symbol_suffix: ".NS" (informational -- symbols already include it)
        - total_stocks: count of constituents
        - fetched_at: ISO timestamp

    Raises:
        NseFetchError: If the NSE India API is unreachable or returns bad data.
    """
    cached = load_cached_universe(CACHE_NAME)
    if cached is not None:
        return cached

    nse_constituents = scrape_nifty500_shariah()

    stocks = [{**h, "symbol": h["symbol"] + NS_SUFFIX} for h in nse_constituents]

    logger.info(
        f"NiftyShariah500: fetched {len(stocks)} constituents from Nifty 500 Shariah"
    )

    result = {
        "stocks": stocks,
        "source": "nifty_500_shariah",
        "symbol_suffix": NS_SUFFIX,
        "total_stocks": len(stocks),
        "fetched_at": datetime.now(UTC).isoformat(),
    }
    save_universe_cache(CACHE_NAME, result)
    return result


def get_nifty_shariah_500_symbols(
    shutdown_event: threading.Event | None = None,
) -> list[str]:
    """Get the list of NiftyShariah500 stock symbols (with .NS suffix).

    Convenience function for use by training pipelines and allocation.

    Args:
        shutdown_event: Reserved for future cancellation support.

    Returns:
        List of ~210 yfinance-ready NSE stock symbols (e.g., 'RELIANCE.NS').
    """
    universe = get_nifty_shariah_500_universe(shutdown_event=shutdown_event)
    return [s["symbol"] for s in universe["stocks"]]

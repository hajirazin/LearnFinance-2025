"""Fetch tradable US equity symbols from the Alpaca paper trading API.

Used by the Halal_New universe to filter scraped ETF holdings down to
only symbols that are actually tradable on Alpaca.
"""

import logging

import httpx

from brain_api.core.config import get_alpaca_api_key, get_alpaca_api_secret

logger = logging.getLogger(__name__)

ALPACA_PAPER_API_URL = "https://paper-api.alpaca.markets/v2/assets"


def fetch_alpaca_tradable_symbols() -> set[str]:
    """Fetch all tradable US equity symbols from Alpaca.

    Calls GET /v2/assets with status=active and asset_class=us_equity,
    then filters to only those marked as tradable.

    Returns:
        Set of tradable ticker symbols.

    Raises:
        RuntimeError: If Alpaca API keys are not configured.
        httpx.HTTPStatusError: If the API returns an error status.
    """
    api_key = get_alpaca_api_key()
    api_secret = get_alpaca_api_secret()
    if not api_key or not api_secret:
        raise RuntimeError(
            "ALPACA_API_KEY and ALPACA_API_SECRET must be set in environment"
        )

    logger.info("Fetching tradable assets from Alpaca API")

    resp = httpx.get(
        ALPACA_PAPER_API_URL,
        headers={
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret,
        },
        params={"status": "active", "asset_class": "us_equity"},
        timeout=60,
    )
    resp.raise_for_status()
    all_assets = resp.json()

    tradable_symbols = {a["symbol"] for a in all_assets if a.get("tradable")}
    logger.info(
        f"Alpaca: {len(all_assets)} active US equities, "
        f"{len(tradable_symbols)} tradable"
    )
    return tradable_symbols

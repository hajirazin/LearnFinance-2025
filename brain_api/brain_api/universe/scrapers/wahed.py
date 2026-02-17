"""Fetch holdings from Wahed ETFs via public Google Sheets CSV export.

Supports HLAL and UMMA. The sheets are maintained by Wahed and contain
daily holdings with columns: Date, Account, StockTicker, CUSIP,
SecurityName, Shares, Price, MarketValue, Weightings, ...
"""

import csv
import io
import logging

import requests

logger = logging.getLogger(__name__)

# Public Google Sheet IDs for Wahed ETF holdings.
WAHED_SHEET_IDS: dict[str, str] = {
    "hlal": "1UC1Bk67bGuYsos_i8y_HQpNoHpVHAvqf71MbgrafJOQ",
    "umma": "1kACYezLTfiN5dWMrM02GL2uQWsYTj2nqVTejp6hJp2k",
}


def scrape_wahed(etf_slug: str) -> list[dict]:
    """Fetch full holdings from a Wahed ETF Google Sheet.

    Downloads the sheet as CSV and parses ticker, name, and weighting.
    Foreign tickers with exchange suffixes (e.g. "005930 KS", "ASML NA")
    are stripped to just the first token so the Alpaca tradable filter
    can handle them downstream.

    Args:
        etf_slug: Lowercase ETF identifier (e.g. "hlal", "umma").

    Returns:
        List of dicts with keys: symbol, name, weight.
        Empty list if the sheet cannot be fetched or parsed.

    Raises:
        KeyError: If etf_slug is not in WAHED_SHEET_IDS.
    """
    sheet_id = WAHED_SHEET_IDS[etf_slug]
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    logger.info(f"Fetching Wahed {etf_slug.upper()} holdings from Google Sheets")

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    reader = csv.DictReader(io.StringIO(resp.text))
    holdings: list[dict] = []
    seen_tickers: set[str] = set()

    for row in reader:
        ticker = row.get("StockTicker", "").strip()
        if not ticker or ticker == "Cash&Other":
            continue

        # Strip exchange suffixes for foreign tickers (e.g. "ASML NA" -> "ASML")
        symbol = ticker.split()[0] if " " in ticker else ticker

        name = row.get("SecurityName", "").strip()
        weight_text = row.get("Weightings", "0").strip().rstrip("%")
        try:
            weight = float(weight_text)
        except ValueError:
            weight = 0.0

        # Deduplicate (sheets may have multiple date rows for same ticker)
        if symbol in seen_tickers:
            continue
        seen_tickers.add(symbol)

        holdings.append({"symbol": symbol, "name": name, "weight": weight})

    logger.info(f"[{etf_slug.upper()}] Fetched {len(holdings)} holdings")
    return holdings

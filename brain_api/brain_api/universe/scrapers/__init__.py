"""Scrapers for fetching ETF holdings and tradable asset data.

Sub-modules:
    sp_funds: Scrape holdings from sp-funds.com (SPUS, SPTE, SPWO)
    wahed:    Fetch holdings from Wahed Google Sheets (HLAL, UMMA)
    alpaca:   Fetch tradable US equity symbols from Alpaca API
    nse:      Fetch Nifty 500 Shariah constituents from NSE India JSON API
"""

from brain_api.universe.scrapers.alpaca import fetch_alpaca_tradable_symbols
from brain_api.universe.scrapers.nse import NseFetchError, scrape_nifty500_shariah
from brain_api.universe.scrapers.sp_funds import scrape_sp_funds
from brain_api.universe.scrapers.wahed import scrape_wahed

__all__ = [
    "NseFetchError",
    "fetch_alpaca_tradable_symbols",
    "scrape_nifty500_shariah",
    "scrape_sp_funds",
    "scrape_wahed",
]

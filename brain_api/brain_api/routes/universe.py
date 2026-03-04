"""Universe endpoints for stock universe data."""

import logging

from fastapi import APIRouter, HTTPException

from brain_api.universe import (
    get_halal_filtered_universe,
    get_halal_india_universe,
    get_halal_new_universe,
    get_halal_universe,
    get_nifty_shariah_500_universe,
)
from brain_api.universe.scrapers.nse import NseFetchError

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/halal")
def get_halal_stocks() -> dict:
    """Get the halal stock universe.

    Returns deduplicated union of top holdings from halal ETFs (SPUS, HLAL, SPTE).
    Each stock includes symbol, name, max weight across ETFs, and which ETFs hold it.

    First call of each month fetches live ETF holdings via yfinance.
    Subsequent calls in the same month are served from cache.
    """
    return get_halal_universe()


@router.get("/halal_new")
def get_halal_new_stocks() -> dict:
    """Get the Halal_New stock universe.

    Scrapes all holdings from 5 halal ETFs (SPUS, SPTE, SPWO, HLAL, UMMA),
    merges and deduplicates, then filters to only Alpaca-tradable symbols.
    Returns ~410 stocks (larger than the original halal universe of ~14).

    First call of each month scrapes live data from SP Funds, Wahed, and Alpaca.
    Subsequent calls in the same month are served from cache.
    """
    return get_halal_new_universe()


@router.get("/halal_filtered")
def get_halal_filtered_stocks() -> dict:
    """Get the Halal_Filtered stock universe (top 15 by PatchTST predicted return).

    Takes the halal_new base (~410 stocks), runs PatchTST inference on all,
    returns top 15 by predicted weekly return.

    First call of each month runs PatchTST inference (may take several minutes).
    Subsequent calls in the same month are served from cache.
    """
    from brain_api.main import shutdown_event

    try:
        return get_halal_filtered_universe(shutdown_event=shutdown_event)
    except ValueError as e:
        logger.error(f"Halal filtered universe build failed: {e}")
        raise HTTPException(status_code=503, detail=str(e)) from e


@router.get("/halal_india")
def get_halal_india_stocks() -> dict:
    """Get the Halal_India stock universe (top 15 by PatchTST predicted return).

    Takes the NiftyShariah500 base (~210 stocks), runs India PatchTST
    inference on all, returns top 15 by predicted weekly return.

    First call of each month runs PatchTST inference.
    Subsequent calls in the same month are served from cache.
    """
    from brain_api.main import shutdown_event

    try:
        return get_halal_india_universe(shutdown_event=shutdown_event)
    except ValueError as e:
        logger.error(f"Halal India universe build failed: {e}")
        raise HTTPException(status_code=503, detail=str(e)) from e
    except NseFetchError as e:
        logger.error(f"Halal India NSE scraper failed: {e}")
        raise HTTPException(status_code=503, detail=str(e)) from e


@router.get("/nifty_shariah_500")
def get_nifty_shariah_500_stocks() -> dict:
    """Get the NiftyShariah500 stock universe (all Nifty 500 Shariah constituents).

    Fetches all ~210 Nifty 500 Shariah Index constituents from NSE India.
    Symbols include .NS suffix for yfinance compatibility.

    First call of each month fetches live data from NSE.
    Subsequent calls in the same month are served from cache.
    """
    try:
        return get_nifty_shariah_500_universe()
    except NseFetchError as e:
        logger.error(f"NiftyShariah500 scraper failed: {e}")
        raise HTTPException(status_code=503, detail=str(e)) from e

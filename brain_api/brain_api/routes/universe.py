"""Universe endpoints for stock universe data."""

import logging

from fastapi import APIRouter, HTTPException

from brain_api.universe import (
    get_halal_filtered_universe,
    get_halal_new_universe,
    get_halal_universe,
)
from brain_api.universe.stock_filter import YFinanceFetchError

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
    """Get the Halal_Filtered stock universe (top 15 factor-scored).

    Takes the halal_new base (~410 stocks), applies junk filter
    (ROE > 0, Price > SMA200, Beta < 2) and factor scoring
    (0.4*Momentum + 0.3*Quality + 0.3*Value), returns top 15.

    First call of each month fetches live yfinance data (~7 minutes).
    Subsequent calls in the same month are served from cache.
    """
    from brain_api.main import shutdown_event

    try:
        return get_halal_filtered_universe(shutdown_event=shutdown_event)
    except YFinanceFetchError as e:
        logger.error(f"Halal filtered universe build failed: {e}")
        raise HTTPException(status_code=503, detail=str(e)) from e

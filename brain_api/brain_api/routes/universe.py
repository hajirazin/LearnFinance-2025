"""Universe endpoints for stock universe data."""

from fastapi import APIRouter

from brain_api.universe import get_halal_new_universe, get_halal_universe

router = APIRouter()


@router.get("/halal")
def get_halal_stocks() -> dict:
    """Get the halal stock universe.

    Returns deduplicated union of top holdings from halal ETFs (SPUS, HLAL, SPTE).
    Each stock includes symbol, name, max weight across ETFs, and which ETFs hold it.
    """
    return get_halal_universe()


@router.get("/halal_new")
def get_halal_new_stocks() -> dict:
    """Get the Halal_New stock universe.

    Scrapes all holdings from 5 halal ETFs (SPUS, SPTE, SPWO, HLAL, UMMA),
    merges and deduplicates, then filters to only Alpaca-tradable symbols.
    Returns ~410 stocks (larger than the original halal universe of ~45).
    """
    return get_halal_new_universe()

"""Universe endpoints for stock universe data."""

from fastapi import APIRouter

from brain_api.universe import get_halal_universe

router = APIRouter()


@router.get("/halal")
def get_halal_stocks() -> dict:
    """Get the halal stock universe.

    Returns deduplicated union of top holdings from halal ETFs (SPUS, HLAL, SPTE).
    Each stock includes symbol, name, max weight across ETFs, and which ETFs hold it.
    """
    return get_halal_universe()



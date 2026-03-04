"""Portfolio allocation endpoints."""

import logging
from datetime import date, timedelta

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from brain_api.core.config import UniverseType
from brain_api.core.hrp import HRPResult, compute_hrp_allocation
from brain_api.core.lstm import load_prices_yfinance
from brain_api.universe import (
    get_halal_filtered_symbols,
    get_halal_india_symbols,
    get_halal_new_symbols,
    get_halal_symbols,
    get_nifty_shariah_500_symbols,
    get_sp500_symbols,
)
from brain_api.universe.scrapers.nse import NseFetchError
from brain_api.universe.stock_filter import YFinanceFetchError

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Request / Response models
# ============================================================================


class HRPAllocationRequest(BaseModel):
    """Request model for HRP allocation endpoint."""

    universe: UniverseType = Field(
        ...,
        description="Stock universe to allocate (e.g. 'halal_filtered', 'halal_india')",
    )
    lookback_days: int = Field(
        252,
        ge=60,
        le=504,
        description="Number of trading days for return calculation (60-504, default 252 = 1 year)",
    )
    as_of_date: str | None = Field(
        None,
        description="Reference date (YYYY-MM-DD). Defaults to today.",
    )


class HRPAllocationResponse(BaseModel):
    """Response model for HRP allocation endpoint."""

    universe: str = Field(
        ...,
        description="Universe used for allocation",
    )
    percentage_weights: dict[str, float] = Field(
        ...,
        description="Target portfolio weights as percentages (sum to 100)",
    )
    symbols_used: int = Field(
        ...,
        description="Number of symbols included in allocation",
    )
    symbols_excluded: list[str] = Field(
        ...,
        description="Symbols excluded due to insufficient data",
    )
    lookback_days: int = Field(
        ...,
        description="Trading days used for return calculation",
    )
    as_of_date: str = Field(
        ...,
        description="Reference date (YYYY-MM-DD)",
    )


def get_as_of_date(request: HRPAllocationRequest) -> date:
    """Get the as-of date from request or default to today."""
    if request.as_of_date:
        return date.fromisoformat(request.as_of_date)
    return date.today()


# Type alias for price loader dependency
PriceLoader = type(load_prices_yfinance)


def get_price_loader() -> PriceLoader:
    """Get the price loading function."""
    return load_prices_yfinance


def _resolve_universe_symbols(universe: UniverseType) -> list[str]:
    """Resolve a UniverseType enum to a list of yfinance-compatible symbols.

    India universes (HALAL_INDIA, NIFTY_SHARIAH_500) already include .NS
    suffix from the universe level, so no transformation is needed here.

    Raises:
        ValueError: If the universe type is not supported.
    """
    resolvers: dict[UniverseType, callable] = {
        UniverseType.HALAL: get_halal_symbols,
        UniverseType.HALAL_NEW: get_halal_new_symbols,
        UniverseType.HALAL_FILTERED: get_halal_filtered_symbols,
        UniverseType.HALAL_INDIA: get_halal_india_symbols,
        UniverseType.NIFTY_SHARIAH_500: get_nifty_shariah_500_symbols,
        UniverseType.SP500: get_sp500_symbols,
    }

    resolver = resolvers.get(universe)
    if resolver is None:
        raise ValueError(f"No symbol resolver for universe '{universe.value}'")

    return resolver()


# ============================================================================
# Endpoint
# ============================================================================


@router.post("/hrp", response_model=HRPAllocationResponse)
def allocate_hrp(
    request: HRPAllocationRequest,
    price_loader: PriceLoader = Depends(get_price_loader),
) -> HRPAllocationResponse:
    """Compute HRP portfolio allocation for a given universe.

    Hierarchical Risk Parity (HRP) allocates weights based on the
    covariance structure of asset returns, using hierarchical clustering
    to group similar assets and recursive bisection to assign weights.

    The algorithm (Lopez de Prado, 2016):
    1. Fetch price history for all symbols in the requested universe
    2. Compute correlation matrix from daily returns
    3. Convert correlation to distance matrix
    4. Hierarchical clustering to group similar assets
    5. Recursive bisection to allocate weights by inverse variance

    Returns percentage weights that sum to 100 (e.g., AAPL: 8.2 means 8.2%).
    """
    try:
        symbols = _resolve_universe_symbols(request.universe)
    except (YFinanceFetchError, NseFetchError, ValueError) as e:
        logger.error(f"HRP universe resolution failed for '{request.universe}': {e}")
        raise HTTPException(status_code=503, detail=str(e)) from e

    as_of = get_as_of_date(request)

    buffer_days = request.lookback_days + 30
    start_date = as_of - timedelta(days=int(buffer_days * 1.5))
    end_date = as_of

    prices = price_loader(symbols, start_date, end_date)

    result: HRPResult = compute_hrp_allocation(
        prices=prices,
        lookback_days=request.lookback_days,
        min_data_days=60,
        as_of_date=as_of,
    )

    if not result.percentage_weights:
        raise HTTPException(
            status_code=400,
            detail=f"No symbols have sufficient data. Excluded: {result.symbols_excluded}",
        )

    sorted_weights = dict(
        sorted(
            result.percentage_weights.items(),
            key=lambda x: x[1],
            reverse=True,
        )
    )

    return HRPAllocationResponse(
        universe=request.universe.value,
        percentage_weights=sorted_weights,
        symbols_used=len(result.symbols_used),
        symbols_excluded=result.symbols_excluded,
        lookback_days=result.lookback_days,
        as_of_date=result.as_of_date,
    )

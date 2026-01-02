"""Portfolio allocation endpoints."""

from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from brain_api.core.hrp import HRPResult, compute_hrp_allocation
from brain_api.core.lstm import load_prices_yfinance
from brain_api.universe import get_halal_universe

router = APIRouter()


# ============================================================================
# Request / Response models
# ============================================================================


class HRPAllocationRequest(BaseModel):
    """Request model for HRP allocation endpoint."""

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


# ============================================================================
# Dependency injection for testability
# ============================================================================


def get_halal_symbols() -> list[str]:
    """Get symbols from halal universe."""
    universe = get_halal_universe()
    return [stock["symbol"] for stock in universe["stocks"]]


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


# ============================================================================
# Endpoint
# ============================================================================


@router.post("/hrp", response_model=HRPAllocationResponse)
def allocate_hrp(
    request: HRPAllocationRequest = HRPAllocationRequest(),
    symbols: list[str] = Depends(get_halal_symbols),
    price_loader: PriceLoader = Depends(get_price_loader),
) -> HRPAllocationResponse:
    """Compute HRP portfolio allocation for the halal universe.

    Hierarchical Risk Parity (HRP) allocates weights based on the
    covariance structure of asset returns, using hierarchical clustering
    to group similar assets and recursive bisection to assign weights.

    The algorithm (López de Prado, 2016):
    1. Fetch price history for all halal symbols
    2. Compute correlation matrix from daily returns
    3. Convert correlation to distance matrix
    4. Hierarchical clustering to group similar assets
    5. Recursive bisection to allocate weights by inverse variance

    Returns percentage weights that sum to 100 (e.g., AAPL: 8.2 means 8.2%).

    Args:
        request: HRPAllocationRequest with lookback_days and as_of_date

    Returns:
        HRPAllocationResponse with percentage weights and metadata

    Raises:
        HTTPException 400: if no symbols have sufficient data
    """
    # Get as-of date
    as_of = get_as_of_date(request)

    # Calculate date range for price data
    # Fetch extra buffer for weekends/holidays
    from datetime import timedelta

    buffer_days = request.lookback_days + 30
    start_date = as_of - timedelta(days=int(buffer_days * 1.5))
    end_date = as_of

    # Fetch price data
    prices = price_loader(symbols, start_date, end_date)

    # Compute HRP allocation
    result: HRPResult = compute_hrp_allocation(
        prices=prices,
        lookback_days=request.lookback_days,
        min_data_days=60,
        as_of_date=as_of,
    )

    # Check if we have any valid allocations
    if not result.percentage_weights:
        raise HTTPException(
            status_code=400,
            detail=f"No symbols have sufficient data. Excluded: {result.symbols_excluded}",
        )

    # Sort weights by percentage descending (highest allocation first)
    sorted_weights = dict(
        sorted(
            result.percentage_weights.items(),
            key=lambda x: x[1],
            reverse=True,
        )
    )

    return HRPAllocationResponse(
        percentage_weights=sorted_weights,
        symbols_used=len(result.symbols_used),
        symbols_excluded=result.symbols_excluded,
        lookback_days=result.lookback_days,
        as_of_date=result.as_of_date,
    )


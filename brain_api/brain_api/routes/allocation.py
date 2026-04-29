"""Portfolio allocation endpoints."""

import logging
from datetime import date, timedelta

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from brain_api.core.hrp import HRPResult, compute_hrp_allocation
from brain_api.core.lstm import load_prices_yfinance
from brain_api.core.sticky_selection import (
    SelectionResult,
    select_with_stickiness,
)
from brain_api.storage.sticky_history import (
    StickyHistoryRepository,
    WeightRow,
    get_sticky_history_repo,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Request / Response models
# ============================================================================


class HRPAllocationRequest(BaseModel):
    """Request model for HRP allocation endpoint."""

    symbols: list[str] = Field(
        ...,
        min_length=1,
        description="List of ticker symbols to allocate (e.g. ['AAPL', 'MSFT'] or ['INFY.NS', 'TCS.NS'])",
    )
    lookback_days: int = Field(
        252,
        ge=60,
        le=756,
        description="Number of trading days for return calculation (60-756, default 252 = 1 year)",
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
    symbols = request.symbols
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
        percentage_weights=sorted_weights,
        symbols_used=len(result.symbols_used),
        symbols_excluded=result.symbols_excluded,
        lookback_days=result.lookback_days,
        as_of_date=result.as_of_date,
    )


# ============================================================================
# Sticky-selection (rebalance-band) models
# ============================================================================


class StickyTopNRequest(BaseModel):
    """Request model for /allocation/sticky-top-n.

    The stage 1 result is a full-universe HRP run (e.g. ~410 stocks for
    halal_new). The endpoint persists those weights, looks up last week's
    final selection for the same universe, and returns the top-N for this
    week with sticky retention applied.
    """

    stage1: HRPAllocationResponse = Field(
        ...,
        description="Stage 1 HRP result over the full universe",
    )
    universe: str = Field(
        ...,
        min_length=1,
        description="Universe label, e.g. 'halal_new', 'nifty_shariah_500'",
    )
    year_week: str = Field(
        ...,
        min_length=6,
        max_length=6,
        description="ISO year-week 'YYYYWW' (e.g. '202608')",
    )
    as_of_date: str = Field(
        ...,
        min_length=10,
        max_length=10,
        description="Reference date 'YYYY-MM-DD'",
    )
    run_id: str = Field(
        ...,
        min_length=1,
        description="Run identifier (e.g. 'paper:2026-02-23')",
    )
    top_n: int = Field(
        15,
        ge=1,
        le=100,
        description="Target number of selected symbols",
    )
    stickiness_threshold_pp: float = Field(
        1.0,
        ge=0.0,
        le=50.0,
        description=(
            "Maximum absolute pp move on stage 1 weight that still "
            "qualifies a previously-held stock as sticky"
        ),
    )


class StickyTopNResponse(BaseModel):
    """Response model for /allocation/sticky-top-n."""

    selected: list[str] = Field(
        ...,
        description="Selected symbols, sticky stocks first",
    )
    reasons: dict[str, str] = Field(
        ...,
        description="Per-symbol reason: 'sticky' or 'top_rank'",
    )
    kept_count: int = Field(
        ...,
        ge=0,
        description="How many last-week holdings were retained as sticky",
    )
    fillers_count: int = Field(
        ...,
        ge=0,
        description="How many slots filled with new top-rank stocks",
    )
    evicted_from_previous: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Map of evicted previous-week symbols to reason: "
            "'weight_diff' or 'dropped_from_universe'"
        ),
    )
    previous_year_week_used: str | None = Field(
        None,
        description="Year-week used for the prior comparison, None on cold start",
    )
    universe: str
    year_week: str


class RecordFinalWeightsRequest(BaseModel):
    """Request model for /allocation/record-final-weights.

    Called after Stage 2 HRP runs on the selected symbols. Updates the
    persisted stage 1 rows to include the final stage 2 weight.
    """

    universe: str = Field(..., min_length=1)
    year_week: str = Field(..., min_length=6, max_length=6)
    final_weights_pct: dict[str, float] = Field(
        ...,
        description="Stage 2 HRP weights (in %) keyed by symbol",
    )


class RecordFinalWeightsResponse(BaseModel):
    """Response model for /allocation/record-final-weights."""

    rows_updated: int = Field(..., ge=0)
    universe: str
    year_week: str


# ============================================================================
# Sticky-selection endpoints
# ============================================================================


@router.post("/sticky-top-n", response_model=StickyTopNResponse)
def select_sticky_top_n_endpoint(
    request: StickyTopNRequest,
    repo: StickyHistoryRepository = Depends(get_sticky_history_repo),
) -> StickyTopNResponse:
    """Persist Stage 1 weights and return top-N with sticky retention.

    This is the rebalance-band primitive: stocks held last week whose
    stage 1 weight has moved by less than ``stickiness_threshold_pp`` are
    retained even if their rank fell. Remaining slots are filled by
    descending current weight.

    On a cold start (no prior week for this universe), the endpoint
    falls back to plain top-N by weight, persists the row set, and
    returns ``previous_year_week_used=None``.
    """
    current = request.stage1.percentage_weights

    if not current:
        raise HTTPException(
            status_code=400,
            detail="stage1.percentage_weights must not be empty",
        )

    previous = repo.read_previous_final_set(
        universe=request.universe,
        current_year_week=request.year_week,
    )
    previous_initial = (
        previous.initial_allocation_by_stock if previous is not None else None
    )
    previous_final_set = previous.final_set if previous is not None else None
    previous_year_week_used = previous.year_week if previous is not None else None

    result: SelectionResult = select_with_stickiness(
        current_stage1=current,
        previous_stage1=previous_initial,
        previous_final_set=previous_final_set,
        top_n=request.top_n,
        threshold_pp=request.stickiness_threshold_pp,
    )

    selected_set = set(result.selected)
    sorted_pairs = sorted(
        current.items(),
        key=lambda kv: (-kv[1], kv[0]),
    )
    rows = [
        WeightRow(
            universe=request.universe,
            year_week=request.year_week,
            as_of_date=request.as_of_date,
            stock=symbol,
            stage1_rank=rank,
            initial_allocation_pct=weight,
            final_allocation_pct=None,
            selected_in_final=(symbol in selected_set),
            selection_reason=result.reasons.get(symbol),
            run_id=request.run_id,
        )
        for rank, (symbol, weight) in enumerate(sorted_pairs, start=1)
    ]
    repo.persist_stage1(rows)

    logger.info(
        "[Sticky] %s/%s: kept=%d fillers=%d prev_week=%s evicted=%d",
        request.universe,
        request.year_week,
        result.kept_count,
        result.fillers_count,
        previous_year_week_used,
        len(result.evicted_from_previous),
    )

    return StickyTopNResponse(
        selected=result.selected,
        reasons=result.reasons,
        kept_count=result.kept_count,
        fillers_count=result.fillers_count,
        evicted_from_previous=result.evicted_from_previous,
        previous_year_week_used=previous_year_week_used,
        universe=request.universe,
        year_week=request.year_week,
    )


@router.post("/record-final-weights", response_model=RecordFinalWeightsResponse)
def record_final_weights_endpoint(
    request: RecordFinalWeightsRequest,
    repo: StickyHistoryRepository = Depends(get_sticky_history_repo),
) -> RecordFinalWeightsResponse:
    """Record Stage 2 final weights for the just-completed week.

    Called after Stage 2 HRP runs on the symbols returned by
    /allocation/sticky-top-n. Updates the corresponding stage 1 rows to
    set ``final_allocation_pct`` and ``selected_in_final=1``. Symbols not
    present in the persisted stage 1 set (which would be unusual but
    possible if the workflow constructed the symbol list manually) are
    silently ignored.
    """
    rows_updated = repo.update_final_weights(
        universe=request.universe,
        year_week=request.year_week,
        final=request.final_weights_pct,
    )
    logger.info(
        "[Sticky] Recorded %d final weights for %s/%s",
        rows_updated,
        request.universe,
        request.year_week,
    )
    return RecordFinalWeightsResponse(
        rows_updated=rows_updated,
        universe=request.universe,
        year_week=request.year_week,
    )

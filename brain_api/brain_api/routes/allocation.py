"""Portfolio allocation endpoints.

Pydantic models live in ``allocation_models.py``; this file is route
handlers + dependency wiring only. The two sticky-selection endpoints
(weight-band and rank-band) share their persistence scaffolding via
``persist_stage1_rows`` (in ``brain_api.core.rank_band_orchestration``)
so the math layers stay independent (per AGENTS.md: never share math
across selection policies).

The rank-band endpoint here is a thin HTTP wrapper over
``select_rank_band_with_persistence``; the same orchestration helper is
testable without FastAPI and is exclusively for the **two-stage**
``stage1_weight_history`` table. Single-stage screening callers (e.g.
``halal_filtered``) use the parallel ``screening_orchestration`` module
against the sibling ``screening_history`` table -- never this endpoint.
"""

import logging
from datetime import date, timedelta

from fastapi import APIRouter, Depends, HTTPException

from brain_api.core.hrp import HRPResult, compute_hrp_allocation
from brain_api.core.lstm import load_prices_yfinance
from brain_api.core.rank_band_orchestration import (
    persist_stage1_rows,
    select_rank_band_with_persistence,
)
from brain_api.core.sticky_selection import (
    SelectionResult,
    select_with_stickiness,
)
from brain_api.routes.allocation_models import (
    HRPAllocationRequest,
    HRPAllocationResponse,
    RankBandTopNRequest,
    RankBandTopNResponse,
    RecordFinalWeightsRequest,
    RecordFinalWeightsResponse,
    StickyTopNRequest,
    StickyTopNResponse,
)
from brain_api.storage.sticky_history import (
    StickyHistoryRepository,
    get_sticky_history_repo,
)

# Re-export response models so existing call sites
# (``from brain_api.routes.allocation import HRPAllocationResponse``)
# keep working.
__all__ = [
    "HRPAllocationRequest",
    "HRPAllocationResponse",
    "RankBandTopNRequest",
    "RankBandTopNResponse",
    "RecordFinalWeightsRequest",
    "RecordFinalWeightsResponse",
    "StickyTopNRequest",
    "StickyTopNResponse",
    "router",
]

logger = logging.getLogger(__name__)

router = APIRouter()


def get_as_of_date(request: HRPAllocationRequest) -> date:
    """Get the as-of date from request or default to today."""
    if request.as_of_date:
        return date.fromisoformat(request.as_of_date)
    return date.today()


PriceLoader = type(load_prices_yfinance)


def get_price_loader() -> PriceLoader:
    """Get the price loading function."""
    return load_prices_yfinance


# ============================================================================
# /allocation/hrp
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
# /allocation/sticky-top-n  (weight-band policy)
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

    persist_stage1_rows(
        repo=repo,
        universe=request.universe,
        year_week=request.year_week,
        as_of_date=request.as_of_date,
        run_id=request.run_id,
        signals=current,
        selected_set=set(result.selected),
        selection_reasons=result.reasons,
        column="initial_allocation_pct",
    )

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
    /allocation/sticky-top-n or /allocation/rank-band-top-n. Updates
    the corresponding stage 1 rows to set ``final_allocation_pct`` and
    ``selected_in_final=1``. Symbols not present in the persisted
    stage 1 set (which would be unusual but possible if the workflow
    constructed the symbol list manually) are silently ignored.
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


# ============================================================================
# /allocation/rank-band-top-n  (rank-band policy)
# ============================================================================


@router.post("/rank-band-top-n", response_model=RankBandTopNResponse)
def select_rank_band_top_n_endpoint(
    request: RankBandTopNRequest,
    repo: StickyHistoryRepository = Depends(get_sticky_history_repo),
) -> RankBandTopNResponse:
    """Persist current scores and return top-N with rank-band stickiness.

    Asymmetric rank-band turnover damper for ranked-alpha screening:
    a previously-held stock is retained as long as its current rank is
    inside ``hold_threshold`` (``K_hold``), even if it slipped out of
    the entry zone ``top_n`` (``K_in``). Remaining slots are filled
    from this week's highest-ranked non-held names.

    The persisted ``signal_score`` column stores the raw numeric
    signal (e.g. PatchTST predicted weekly return %) for audit;
    ``initial_allocation_pct`` stays NULL because rank-band selection
    is not weight-based and that column's documented unit is "Stage 1
    HRP weight in %" (Σ ≈ 100% across the universe).

    On a cold start (no prior week for this universe), the endpoint
    falls back to plain top-N by current score, persists the row set,
    and returns ``previous_year_week_used=None``.
    """
    current = request.current_scores

    if not current:
        raise HTTPException(
            status_code=400,
            detail="current_scores must not be empty",
        )

    if request.hold_threshold < request.top_n:
        raise HTTPException(
            status_code=422,
            detail=(
                f"hold_threshold ({request.hold_threshold}) must be >= "
                f"top_n ({request.top_n})"
            ),
        )

    try:
        result, previous_year_week_used = select_rank_band_with_persistence(
            repo=repo,
            universe=request.universe,
            year_week=request.year_week,
            as_of_date=request.as_of_date,
            run_id=request.run_id,
            current_scores=current,
            top_n=request.top_n,
            hold_threshold=request.hold_threshold,
        )
    except ValueError as exc:
        # Non-finite scores or other validation failures must surface
        # as a 422 rather than a generic 500; they indicate caller
        # input that violates the rank-band contract.
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    logger.info(
        "[RankBand] %s/%s: top_n=%d hold=%d kept=%d fillers=%d prev_week=%s evicted=%d",
        request.universe,
        request.year_week,
        request.top_n,
        request.hold_threshold,
        result.kept_count,
        result.fillers_count,
        previous_year_week_used,
        len(result.evicted_from_previous),
    )

    return RankBandTopNResponse(
        selected=result.selected,
        reasons=result.reasons,
        kept_count=result.kept_count,
        fillers_count=result.fillers_count,
        evicted_from_previous=result.evicted_from_previous,
        previous_year_week_used=previous_year_week_used,
        universe=request.universe,
        year_week=request.year_week,
        top_n=request.top_n,
        hold_threshold=request.hold_threshold,
    )

"""Pydantic models for the /allocation router.

Split out of ``brain_api/routes/allocation.py`` to keep the route
handler file under the workspace's 600-line limit and to give
sticky-selection / rank-band schemas a clean import surface for
unit tests and other consumers.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class HRPAllocationRequest(BaseModel):
    """Request model for HRP allocation endpoint."""

    symbols: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "List of ticker symbols to allocate (e.g. ['AAPL', 'MSFT'] "
            "or ['INFY.NS', 'TCS.NS'])"
        ),
    )
    lookback_days: int = Field(
        252,
        ge=60,
        le=756,
        description=(
            "Number of trading days for return calculation "
            "(60-756, default 252 = 1 year)"
        ),
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


class StickyTopNRequest(BaseModel):
    """Request model for /allocation/sticky-top-n.

    The stage 1 result is a full-universe HRP run (e.g. ~410 stocks for
    halal_new). The endpoint persists those weights, looks up last
    week's final selection for the same universe, and returns the
    top-N for this week with sticky retention applied.
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


class RankBandTopNRequest(BaseModel):
    """Request model for /allocation/rank-band-top-n.

    Asymmetric rank-band variant of sticky selection: the caller passes
    a pre-computed numeric signal (e.g. PatchTST predicted weekly
    return %) for every symbol in scope, and the endpoint applies a
    ``K_in`` / ``K_hold`` rank-band turnover damper before returning
    the chosen ``top_n`` symbols. ``current_scores`` is treated only
    as a *ranking* signal -- its absolute scale does not affect
    selection, only ordering.
    """

    current_scores: dict[str, float] = Field(
        ...,
        description=(
            "Symbol -> numeric signal (higher is better, e.g. PatchTST "
            "predicted weekly return %). Must be non-empty."
        ),
    )
    universe: str = Field(
        ...,
        min_length=1,
        description=(
            "Sticky-history partition key, e.g. 'halal_new_alpha' for "
            "the alpha-screened halal_new pipeline. Must differ from "
            "any other strategy's label to keep sticky_history rows "
            "isolated. See brain_api.core.strategy_partitions."
        ),
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
        description="Entry threshold K_in: target count of selected symbols",
    )
    hold_threshold: int = Field(
        30,
        ge=1,
        le=500,
        description=(
            "Hold threshold K_hold: a previously-held stock retains "
            "sticky status as long as its current rank is <= this value. "
            "Must be >= top_n; widening K_hold reduces turnover."
        ),
    )


class RankBandTopNResponse(BaseModel):
    """Response model for /allocation/rank-band-top-n.

    Mirrors :class:`StickyTopNResponse` plus the rank-band-specific
    eviction reasons (``rank_out_of_hold`` instead of ``weight_diff``).
    """

    selected: list[str] = Field(
        ...,
        description="Selected symbols, sticky stocks first (each block in rank asc)",
    )
    reasons: dict[str, str] = Field(
        ...,
        description="Per-symbol reason: 'sticky' or 'top_rank'",
    )
    kept_count: int = Field(..., ge=0)
    fillers_count: int = Field(..., ge=0)
    evicted_from_previous: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Map of evicted previous-week symbols to reason: "
            "'rank_out_of_hold' or 'dropped_from_universe'"
        ),
    )
    previous_year_week_used: str | None = Field(
        None,
        description="Year-week used for the prior comparison, None on cold start",
    )
    universe: str
    year_week: str
    top_n: int
    hold_threshold: int

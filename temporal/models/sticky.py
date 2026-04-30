"""Pydantic models for the sticky-selection (rebalance-band) layer.

Mirror of the brain_api response shapes for /allocation/sticky-top-n,
/allocation/rank-band-top-n, and /allocation/record-final-weights.
The PatchTST batch-score model lives in ``models.alpha_screen`` --
sticky-selection only *consumes* alpha scores, it does not own them.
"""

from pydantic import BaseModel


class StickyTopNResponse(BaseModel):
    """Response from POST /allocation/sticky-top-n."""

    selected: list[str]
    reasons: dict[str, str]
    kept_count: int
    fillers_count: int
    evicted_from_previous: dict[str, str] = {}
    previous_year_week_used: str | None = None
    universe: str
    year_week: str


class RecordFinalWeightsResponse(BaseModel):
    """Response from POST /allocation/record-final-weights."""

    rows_updated: int
    universe: str
    year_week: str


class RankBandTopNResponse(BaseModel):
    """Response from POST /allocation/rank-band-top-n.

    Mirrors :class:`StickyTopNResponse` plus the rank-band-specific
    knobs (``top_n``, ``hold_threshold``) and eviction reasons (e.g.
    ``rank_out_of_hold`` instead of ``weight_diff``).
    """

    selected: list[str]
    reasons: dict[str, str]
    kept_count: int
    fillers_count: int
    evicted_from_previous: dict[str, str] = {}
    previous_year_week_used: str | None = None
    universe: str
    year_week: str
    top_n: int
    hold_threshold: int

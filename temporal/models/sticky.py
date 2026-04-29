"""Pydantic models for the sticky-selection (rebalance-band) layer.

Mirror of the brain_api response shapes for /allocation/sticky-top-n
and /allocation/record-final-weights. Kept in its own module to avoid
bloating ``forecast_email.py``.
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

"""Pydantic models for the alpha-screen pipeline.

Mirrors brain_api response shapes for the PatchTST batch scoring
step that feeds rank-band sticky selection. Kept separate from
``models/sticky.py`` because the alpha screen is a *forecast batch*
bounded context, not a sticky-selection bounded context -- the
sticky model only consumes its output.
"""

from pydantic import BaseModel


class PatchTSTBatchScores(BaseModel):
    """Result of running PatchTST inference across a fixed symbol list.

    Used by the alpha-screening step of ``USAlphaHRPWorkflow``: a fresh
    weekly batch over the full ``halal_new`` universe whose only output
    is ``{symbol -> predicted_weekly_return_pct}`` plus week-boundary
    metadata for downstream summary/email rendering. Symbols whose
    PatchTST prediction is ``None`` (insufficient history, etc.) are
    excluded from ``scores`` so the rank-band selector receives only
    finite, comparable values.
    """

    scores: dict[str, float]
    model_version: str
    as_of_date: str
    target_week_start: str | None = None
    target_week_end: str | None = None
    requested_count: int
    predicted_count: int
    excluded_symbols: list[str] = []

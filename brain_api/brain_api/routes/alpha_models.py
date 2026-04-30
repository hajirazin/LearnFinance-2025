"""Shared Pydantic models for the alpha-screen pipeline.

Used by the LLM summary route, the email-rendering route, and any
future consumer of "Stage 1 PatchTST top-K" rows. Keeping a single
definition prevents the model from drifting between routes and
keeps ``rank`` semantics aligned with
``brain_api.core.sticky_selection.rank_by_score``.
"""

from __future__ import annotations

from pydantic import BaseModel


class AlphaScoreItem(BaseModel):
    """One row of the Stage 1 PatchTST top-K table.

    ``rank`` is 1-indexed by ``score`` descending with symbol-asc
    tie-break -- the same ordering ``select_with_rank_band`` uses, so
    ranks shown in downstream consumers (LLM prompt, email) match the
    ranks the selection function actually saw.
    """

    symbol: str
    score: float  # PatchTST predicted_weekly_return_pct
    rank: int

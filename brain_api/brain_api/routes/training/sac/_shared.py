"""Shared constants and request schemas for SAC training endpoints.

Split out of the original ``routes/training/sac.py`` so the file stays
under the 600-line limit (AGENTS.md rule). The full-retrain and
preflight endpoints live in sibling modules and import what they need
directly from the relevant ``brain_api.core`` modules so that mocks
applied at the route module level (``monkeypatch.setattr``) hit the
same lookup the route function makes at call time.
"""

import logging

from pydantic import BaseModel, Field

from brain_api.core.model_buckets import ModelType, list_universes_for

logger = logging.getLogger(__name__)

MIN_PROMOTION_CAGR = 0.12
"""Legacy SAC promotion floor.

Kept here through Phase A (mechanical move only). Phase E renames this
to ``SAC_PROMOTION_CAGR_FLOOR`` and moves it to
``brain_api.core.sac.promotion``.
"""


def sac_us_allowed_universes() -> frozenset[str]:
    """Universes the US SAC endpoint accepts.

    Pulled live from the registry so adding a new bucket
    (e.g. ``(SAC, halal)`` for an A/B comparison vs ``halal_filtered``)
    becomes a one-line addition in ``model_buckets.py`` -- no edit
    needed here. Future India SAC will get its own router file with
    its own allowlist.
    """
    return list_universes_for(ModelType.SAC)


class SACTrainRequest(BaseModel):
    """Body for ``POST /train/sac/full``."""

    universe: str = Field(
        default="halal_filtered",
        description=(
            "Universe to train on. Two parallel workflows can hit this "
            "endpoint with different ``universe`` values for an A/B "
            "comparison; each writes to its own bucket."
        ),
    )
    force: bool = Field(
        default=False,
        description=(
            "When False (default), if the bucket's current model was trained "
            "on the exact same symbol set, return 200 with the current "
            "model's metadata instead of starting a new training job. "
            "Set True to bypass this short-circuit and force retraining."
        ),
    )

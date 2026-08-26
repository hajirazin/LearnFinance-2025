"""Generic statistical primitives for news adapters. No RL imports."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime

from brain_api.news.errors import NewsError
from brain_api.news.models import CONFIDENCE_RECENCY_TAU_HOURS


def confidence_recency_weighted_mean(
    scores: Sequence[float],
    confidences: Sequence[float],
    created_ats: Sequence[datetime],
    cutoff: datetime,
    tau: float = CONFIDENCE_RECENCY_TAU_HOURS,
) -> float:
    """Return ``sum(score_i * w_i) / sum(w_i)`` with ``w_i = conf * exp(-age_h / tau)``.

    Raises if there is at least one observation but the weight sum is zero.
    """
    if cutoff.tzinfo is None:
        raise NewsError("cutoff must be timezone-aware")
    n = len(scores)
    if n != len(confidences) or n != len(created_ats):
        raise NewsError("scores, confidences, and created_ats must have equal length")
    if n == 0:
        return 0.0
    if tau <= 0 or not math.isfinite(tau):
        raise NewsError("tau must be finite and positive")

    weighted = 0.0
    weight_sum = 0.0
    for score, confidence, created_at in zip(
        scores, confidences, created_ats, strict=True
    ):
        if created_at.tzinfo is None:
            raise NewsError("created_at must be timezone-aware")
        if not (
            math.isfinite(score)
            and math.isfinite(confidence)
            and 0.0 <= confidence <= 1.0
        ):
            raise NewsError("score/confidence must be finite; confidence in [0, 1]")
        age_hours = (
            cutoff - created_at.astimezone(cutoff.tzinfo)
        ).total_seconds() / 3600.0
        weight = float(confidence) * math.exp(-age_hours / tau)
        weighted += float(score) * weight
        weight_sum += weight
    if weight_sum == 0.0:
        raise NewsError("weight sum is zero for a non-empty news event set")
    return weighted / weight_sum


def population_std(values: Sequence[float]) -> float:
    """Unweighted population standard deviation. Empty or singleton → 0."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((value - mean) ** 2 for value in values) / n
    return math.sqrt(variance)

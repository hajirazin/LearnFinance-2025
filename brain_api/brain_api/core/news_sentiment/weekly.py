"""Point-in-time weekly news observations shared by SAC training/inference."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date


class NewsObservationError(ValueError):
    """Raised when a weekly news window is missing or provider-unchecked."""


@dataclass(frozen=True)
class DailyNewsObservation:
    """One provider-checked daily aggregate."""

    observation_date: date
    sentiment_score: float
    article_count: int
    avg_confidence: float


@dataclass(frozen=True)
class WeeklyNewsObservation:
    """News state known by a weekly decision cutoff."""

    as_of_date: date
    sentiment_score: float
    coverage: float
    article_count: int
    confirmed_no_articles: bool


def aggregate_weekly_news_observation(
    observations: Iterable[DailyNewsObservation],
    as_of_date: date,
    tau_days: float = 7.0,
) -> WeeklyNewsObservation:
    """Aggregate checked daily rows with confidence and recency weighting.

    ``weight_j = article_count_j * avg_confidence_j * exp(-age_days / 7)``.
    A zero-article window is neutral only when explicit checked rows exist.
    Missing/unchecked windows must be rejected by the caller before aggregation.
    """
    rows = list(observations)
    if not rows:
        raise NewsObservationError(
            f"No provider-checked news observations through {as_of_date}"
        )
    if tau_days <= 0 or not math.isfinite(tau_days):
        raise NewsObservationError("tau_days must be finite and positive")

    total_articles = 0
    weighted_sentiment = 0.0
    total_weight = 0.0
    for row in rows:
        if row.observation_date > as_of_date:
            raise NewsObservationError("News observation is after the decision cutoff")
        if row.article_count < 0:
            raise NewsObservationError("article_count cannot be negative")
        if not (
            math.isfinite(row.sentiment_score)
            and math.isfinite(row.avg_confidence)
            and 0.0 <= row.avg_confidence <= 1.0
        ):
            raise NewsObservationError("News observation contains invalid values")
        if row.article_count > 0 and row.avg_confidence <= 0:
            raise NewsObservationError(
                "Positive article_count requires positive avg_confidence"
            )

        age_days = (as_of_date - row.observation_date).days
        weight = row.article_count * row.avg_confidence * math.exp(-age_days / tau_days)
        total_articles += row.article_count
        weighted_sentiment += row.sentiment_score * weight
        total_weight += weight

    if total_articles == 0:
        return WeeklyNewsObservation(
            as_of_date=as_of_date,
            sentiment_score=0.0,
            coverage=0.0,
            article_count=0,
            confirmed_no_articles=True,
        )
    if total_weight <= 0:
        raise NewsObservationError("Article-bearing news window has zero total weight")

    return WeeklyNewsObservation(
        as_of_date=as_of_date,
        sentiment_score=weighted_sentiment / total_weight,
        coverage=min(total_articles / 3.0, 1.0),
        article_count=total_articles,
        confirmed_no_articles=False,
    )

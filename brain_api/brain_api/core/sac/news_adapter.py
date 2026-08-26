"""SAC news adapter: one finite sentiment scalar per symbol."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime

from brain_api.news.aggregation import confidence_recency_weighted_mean
from brain_api.news.models import CONFIDENCE_RECENCY_TAU_HOURS, NewsEvent


class SACNewsAdapterError(ValueError):
    """Raised when SAC cannot build a news feature from events."""


def build_sac_news_features(
    events_by_symbol: Mapping[str, Sequence[NewsEvent]],
    *,
    cutoff: datetime,
    coverage_status: Mapping[str, str] | None = None,
) -> dict[str, float]:
    """Return one sentiment value per requested symbol.

    Verified empty (no events) returns ``0.0``. Missing coverage must be
    rejected by the caller before this function is invoked.
    """
    features: dict[str, float] = {}
    for symbol, events in events_by_symbol.items():
        if not events:
            status = (coverage_status or {}).get(symbol)
            if status not in (None, "verified_empty", "complete"):
                raise SACNewsAdapterError(
                    f"news coverage for {symbol} is not complete ({status!r})"
                )
            features[symbol] = 0.0
            continue
        features[symbol] = confidence_recency_weighted_mean(
            [event.sentiment_score for event in events],
            [event.confidence for event in events],
            [event.created_at for event in events],
            cutoff,
            tau=CONFIDENCE_RECENCY_TAU_HOURS,
        )
    return features

"""Sentiment aggregation logic."""

from __future__ import annotations

import math
from datetime import UTC, datetime

from brain_api.core.news_sentiment.models import ScoredArticle, SymbolSentiment


def compute_recency_weight(
    published: datetime | None,
    as_of: datetime,
    tau_days: float = 7.0,
) -> float:
    """Compute recency weight for an article.

    Uses exponential decay: weight = exp(-age_days / tau_days)

    Args:
        published: Article publication time (if unknown, returns 0.5)
        as_of: Reference time for computing age
        tau_days: Decay constant in days (default 7 = one week half-life)

    Returns:
        Weight in range (0, 1]
    """
    if published is None:
        # Unknown date: use a middle-ground weight
        return 0.5

    # Compute age in days
    if published.tzinfo is None:
        published = published.replace(tzinfo=UTC)
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=UTC)

    age_seconds = (as_of - published).total_seconds()
    age_days = max(0, age_seconds / 86400)  # Clamp negative ages to 0

    return math.exp(-age_days / tau_days)


def aggregate_symbol_sentiment(
    symbol: str,
    scored_articles: list[ScoredArticle],
    as_of: datetime,
    tau_days: float = 7.0,
    min_articles: int = 3,
) -> SymbolSentiment:
    """Compute aggregated sentiment for a symbol from scored articles.

    Uses recency-weighted average of article scores.

    Args:
        symbol: Stock ticker symbol
        scored_articles: List of articles with FinBERT scores
        as_of: Reference time for recency weighting
        tau_days: Decay constant for recency weighting
        min_articles: Minimum articles needed for confidence

    Returns:
        SymbolSentiment with aggregated score and metadata
    """
    if not scored_articles:
        return SymbolSentiment(
            symbol=symbol,
            article_count_fetched=0,
            article_count_used=0,
            sentiment_score=0.0,
            insufficient_news=True,
            scored_articles=[],
        )

    # Compute weighted average
    total_weight = 0.0
    weighted_sum = 0.0

    for sa in scored_articles:
        weight = compute_recency_weight(sa.article.published, as_of, tau_days)
        weighted_sum += weight * sa.sentiment.score
        total_weight += weight

    sentiment_score = weighted_sum / total_weight if total_weight > 0 else 0.0

    return SymbolSentiment(
        symbol=symbol,
        article_count_fetched=len(scored_articles),
        article_count_used=len(scored_articles),
        sentiment_score=round(sentiment_score, 4),
        insufficient_news=len(scored_articles) < min_articles,
        scored_articles=scored_articles,
    )



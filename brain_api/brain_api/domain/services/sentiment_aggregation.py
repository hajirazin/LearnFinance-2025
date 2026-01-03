"""Sentiment aggregation domain service.

Pure functions for computing sentiment scores with recency weighting.
"""

import math
from datetime import UTC, datetime

from brain_api.domain.entities.sentiment import (
    FinBERTResult,
    ScoredArticle,
    SymbolSentiment,
)


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
    """Aggregate sentiment from multiple scored articles for a symbol.

    Uses recency-weighted average of article scores.

    Args:
        symbol: Stock ticker symbol
        scored_articles: List of articles with FinBERT scores
        as_of: Reference time for recency weighting
        tau_days: Decay constant for exponential weighting
        min_articles: Minimum articles to consider sufficient

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

    # Filter out articles with very weak sentiment
    usable_articles = [
        sa
        for sa in scored_articles
        if abs(sa.finbert.article_score) >= 0.01  # Ignore nearly neutral
    ]

    if len(usable_articles) < min_articles:
        # Not enough articles - return neutral with flag
        return SymbolSentiment(
            symbol=symbol,
            article_count_fetched=len(scored_articles),
            article_count_used=len(usable_articles),
            sentiment_score=0.0,
            insufficient_news=True,
            scored_articles=scored_articles,
        )

    # Compute recency-weighted average
    total_weight = 0.0
    weighted_sum = 0.0

    for sa in usable_articles:
        weight = compute_recency_weight(sa.article.published, as_of, tau_days)
        weighted_sum += weight * sa.finbert.article_score
        total_weight += weight

    sentiment_score = weighted_sum / total_weight if total_weight > 0 else 0.0

    return SymbolSentiment(
        symbol=symbol,
        article_count_fetched=len(scored_articles),
        article_count_used=len(usable_articles),
        sentiment_score=round(sentiment_score, 4),
        insufficient_news=False,
        scored_articles=scored_articles,
    )


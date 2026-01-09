"""Main news sentiment processing pipeline."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from brain_api.core.news_sentiment.aggregation import aggregate_symbol_sentiment
from brain_api.core.news_sentiment.models import (
    NewsSentimentResult,
    ScoredArticle,
    SymbolSentiment,
)
from brain_api.core.news_sentiment.persistence import (
    load_cached_features,
    load_cached_symbol,
    save_features,
    save_raw_news,
)

if TYPE_CHECKING:
    from brain_api.core.news_sentiment.protocols import NewsFetcher, SentimentScorer


def process_symbol_news(
    symbol: str,
    fetcher: NewsFetcher,
    scorer: SentimentScorer,
    max_articles: int,
    as_of: datetime,
    tau_days: float = 7.0,
) -> SymbolSentiment:
    """Fetch and score news for a single symbol.

    Args:
        symbol: Stock ticker symbol
        fetcher: NewsFetcher implementation
        scorer: SentimentScorer implementation
        max_articles: Maximum articles to fetch
        as_of: Reference time for recency weighting
        tau_days: Decay constant for recency weighting

    Returns:
        SymbolSentiment with scored articles and aggregated score
    """
    # Fetch articles
    articles = fetcher.fetch(symbol, max_articles)

    if not articles:
        return SymbolSentiment(
            symbol=symbol,
            article_count_fetched=0,
            article_count_used=0,
            sentiment_score=0.0,
            insufficient_news=True,
            scored_articles=[],
        )

    # Prepare texts for scoring (use title + summary if available)
    texts = []
    for article in articles:
        if article.summary:
            text = f"{article.title}. {article.summary}"
        else:
            text = article.title
        texts.append(text)

    # Score all articles in batch
    sentiment_results = scorer.score_batch(texts)

    # Combine articles with scores
    scored_articles = [
        ScoredArticle(article=article, sentiment=sentiment)
        for article, sentiment in zip(articles, sentiment_results, strict=True)
    ]

    # Aggregate
    return aggregate_symbol_sentiment(
        symbol=symbol,
        scored_articles=scored_articles,
        as_of=as_of,
        tau_days=tau_days,
    )


def process_news_sentiment(
    symbols: list[str],
    fetcher: NewsFetcher,
    scorer: SentimentScorer,
    as_of_date: date,
    max_articles_per_symbol: int = 10,
    run_id: str | None = None,
    attempt: int = 1,
    base_path: Path | None = None,
    tau_days: float = 7.0,
) -> NewsSentimentResult:
    """Process news sentiment for multiple symbols.

    Fetches news, scores with FinBERT, aggregates, and persists results.
    If results already exist for run_id+attempt, returns cached data.

    Args:
        symbols: List of stock ticker symbols
        fetcher: NewsFetcher implementation
        scorer: SentimentScorer implementation
        as_of_date: Reference date for the run
        max_articles_per_symbol: Max articles to fetch per symbol
        run_id: Run identifier (defaults to paper:<as_of_date>)
        attempt: Attempt number (defaults to 1)
        base_path: Base data directory (defaults to data/)
        tau_days: Decay constant for recency weighting

    Returns:
        NewsSentimentResult with all sentiment data
    """
    # Set defaults
    if run_id is None:
        run_id = f"paper:{as_of_date.isoformat()}"
    if base_path is None:
        base_path = Path("data")

    # Check for cached result
    cached = load_cached_features(base_path, run_id, attempt)
    if cached is not None:
        # Return cached data
        return NewsSentimentResult(
            run_id=cached["run_id"],
            attempt=cached["attempt"],
            as_of_date=cached["as_of_date"],
            per_symbol=[SymbolSentiment.from_dict(s) for s in cached["per_symbol"]],
            from_cache=True,
        )

    # Process each symbol
    as_of_datetime = datetime.combine(as_of_date, datetime.min.time(), tzinfo=UTC)
    sentiments: list[SymbolSentiment] = []

    for symbol in symbols:
        # Check for cached symbol data
        cached_symbol = load_cached_symbol(base_path, run_id, attempt, symbol)
        if cached_symbol is not None:
            sentiments.append(cached_symbol)
            continue

        # Process fresh
        sentiment = process_symbol_news(
            symbol=symbol,
            fetcher=fetcher,
            scorer=scorer,
            max_articles=max_articles_per_symbol,
            as_of=as_of_datetime,
            tau_days=tau_days,
        )
        sentiments.append(sentiment)

        # Save raw news immediately
        save_raw_news(base_path, run_id, attempt, symbol, sentiment)

    # Save aggregated features
    save_features(base_path, run_id, attempt, as_of_date.isoformat(), sentiments)

    return NewsSentimentResult(
        run_id=run_id,
        attempt=attempt,
        as_of_date=as_of_date.isoformat(),
        per_symbol=sentiments,
        from_cache=False,
    )



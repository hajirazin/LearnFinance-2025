"""Signal endpoints for feature extraction (news sentiment, etc.)."""

from datetime import date
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from brain_api.core.news_sentiment import (
    FinBERTScorer,
    NewsFetcher,
    NewsSentimentResult,
    SentimentScorer,
    SymbolSentiment,
    YFinanceNewsFetcher,
    process_news_sentiment,
)

router = APIRouter()


# ============================================================================
# Configuration constants
# ============================================================================

MAX_SYMBOLS = 50
MAX_ARTICLES_PER_SYMBOL = 30
MAX_RETURN_TOP_K = 10

DEFAULT_MAX_ARTICLES = 30
DEFAULT_RETURN_TOP_K = 10


# ============================================================================
# Request / Response models
# ============================================================================


class NewsSignalRequest(BaseModel):
    """Request model for news sentiment endpoint."""

    symbols: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_SYMBOLS,
        description=f"List of ticker symbols (1-{MAX_SYMBOLS})",
    )
    as_of_date: str | None = Field(
        None,
        description="Reference date (YYYY-MM-DD). Defaults to today.",
    )
    max_articles_per_symbol: int = Field(
        DEFAULT_MAX_ARTICLES,
        ge=1,
        le=MAX_ARTICLES_PER_SYMBOL,
        description=f"Max articles to fetch per symbol (1-{MAX_ARTICLES_PER_SYMBOL})",
    )
    return_top_k: int = Field(
        DEFAULT_RETURN_TOP_K,
        ge=1,
        le=MAX_RETURN_TOP_K,
        description=f"Number of top articles to return per symbol (1-{MAX_RETURN_TOP_K})",
    )
    run_id: str | None = Field(
        None,
        description="Run identifier. Defaults to paper:<as_of_date>",
    )
    attempt: int = Field(
        1,
        ge=1,
        description="Attempt number for the run",
    )


class ArticleResponse(BaseModel):
    """Article in the API response (subset of stored data)."""

    title: str
    publisher: str
    link: str
    published: str | None
    finbert_label: str
    finbert_p_pos: float
    finbert_p_neg: float
    finbert_p_neu: float
    article_score: float


class SymbolSentimentResponse(BaseModel):
    """Per-symbol sentiment in the API response."""

    symbol: str
    article_count_fetched: int
    article_count_used: int
    sentiment_score: float
    insufficient_news: bool
    top_k_articles: list[ArticleResponse]


class NewsSignalResponse(BaseModel):
    """Response model for news sentiment endpoint."""

    run_id: str
    attempt: int
    as_of_date: str
    from_cache: bool
    per_symbol: list[SymbolSentimentResponse]


# ============================================================================
# Dependency injection for testability
# ============================================================================


def get_news_fetcher() -> NewsFetcher:
    """Get the news fetcher implementation."""
    return YFinanceNewsFetcher()


def get_sentiment_scorer() -> SentimentScorer:
    """Get the sentiment scorer implementation."""
    return FinBERTScorer()


def get_data_base_path() -> Path:
    """Get the base path for data storage."""
    return Path("data")


# ============================================================================
# Helper functions
# ============================================================================


def _symbol_to_response(
    sentiment: SymbolSentiment,
    return_top_k: int,
) -> SymbolSentimentResponse:
    """Convert internal SymbolSentiment to API response format.

    Selects top K articles by article_score (most positive first).
    """
    # Sort articles by article_score descending
    sorted_articles = sorted(
        sentiment.scored_articles,
        key=lambda a: a.finbert.article_score,
        reverse=True,
    )

    # Take top K
    top_articles = sorted_articles[:return_top_k]

    # Convert to response format
    article_responses = [
        ArticleResponse(
            title=sa.article.title,
            publisher=sa.article.publisher,
            link=sa.article.link,
            published=sa.article.published.isoformat() if sa.article.published else None,
            finbert_label=sa.finbert.label,
            finbert_p_pos=sa.finbert.p_pos,
            finbert_p_neg=sa.finbert.p_neg,
            finbert_p_neu=sa.finbert.p_neu,
            article_score=sa.finbert.article_score,
        )
        for sa in top_articles
    ]

    return SymbolSentimentResponse(
        symbol=sentiment.symbol,
        article_count_fetched=sentiment.article_count_fetched,
        article_count_used=sentiment.article_count_used,
        sentiment_score=sentiment.sentiment_score,
        insufficient_news=sentiment.insufficient_news,
        top_k_articles=article_responses,
    )


def _result_to_response(
    result: NewsSentimentResult,
    return_top_k: int,
) -> NewsSignalResponse:
    """Convert internal result to API response format."""
    return NewsSignalResponse(
        run_id=result.run_id,
        attempt=result.attempt,
        as_of_date=result.as_of_date,
        from_cache=result.from_cache,
        per_symbol=[
            _symbol_to_response(s, return_top_k) for s in result.per_symbol
        ],
    )


# ============================================================================
# Endpoint
# ============================================================================


@router.post("/news", response_model=NewsSignalResponse)
def get_news_sentiment(
    request: NewsSignalRequest,
    fetcher: Annotated[NewsFetcher, Depends(get_news_fetcher)],
    scorer: Annotated[SentimentScorer, Depends(get_sentiment_scorer)],
    base_path: Annotated[Path, Depends(get_data_base_path)],
) -> NewsSignalResponse:
    """Get news sentiment scores for the given symbols.

    This endpoint:
    1. Fetches news articles from yfinance for each symbol
    2. Scores each article using FinBERT (financial sentiment model)
    3. Computes a recency-weighted aggregate score per symbol
    4. Persists raw articles and features for audit/training
    5. Returns top K articles per symbol with scores

    If data already exists for the same run_id+attempt, returns cached results
    (idempotent for retries).

    Args:
        request: NewsSignalRequest with symbols and parameters

    Returns:
        NewsSignalResponse with per-symbol sentiment and top articles
    """
    # Parse as-of date
    if request.as_of_date:
        as_of = date.fromisoformat(request.as_of_date)
    else:
        as_of = date.today()

    # Derive run_id if not provided
    run_id = request.run_id
    if run_id is None:
        run_id = f"paper:{as_of.isoformat()}"

    # Ensure return_top_k doesn't exceed max_articles_per_symbol
    return_top_k = min(request.return_top_k, request.max_articles_per_symbol)

    # Process news sentiment (with caching)
    result = process_news_sentiment(
        symbols=request.symbols,
        fetcher=fetcher,
        scorer=scorer,
        as_of_date=as_of,
        max_articles_per_symbol=request.max_articles_per_symbol,
        run_id=run_id,
        attempt=request.attempt,
        base_path=base_path,
    )

    # Convert to response format
    return _result_to_response(result, return_top_k)




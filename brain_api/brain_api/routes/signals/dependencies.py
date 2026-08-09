"""Dependency injection for signal endpoints."""

from pathlib import Path

from brain_api.core.finbert import FinBERTScorer
from brain_api.core.news_sentiment import (
    NewsFetcher,
    SentimentScorer,
    YFinanceNewsFetcher,
)


def get_news_fetcher() -> NewsFetcher:
    """Get the news fetcher implementation."""
    return YFinanceNewsFetcher()


def get_sentiment_scorer() -> SentimentScorer:
    """Get the sentiment scorer implementation."""
    return FinBERTScorer()


def get_data_base_path() -> Path:
    """Get the base path for data storage."""
    return Path("data")


def get_sentiment_parquet_path() -> Path:
    """Get the path to the historical sentiment parquet file.

    The parquet is at project root /data/output/, not brain_api/data/.
    Uses __file__ to get the correct path regardless of working directory.
    """
    # brain_api/brain_api/routes/signals/dependencies.py -> go up 5 levels to project root
    project_root = Path(__file__).parent.parent.parent.parent.parent
    return project_root / "data" / "output" / "daily_sentiment.parquet"

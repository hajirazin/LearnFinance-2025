"""Dependency injection for signal endpoints."""

import os
from pathlib import Path
from typing import Annotated

from fastapi import Depends

from brain_api.core.finbert import FinBERTScorer
from brain_api.core.fundamentals import FundamentalsFetcher
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


def get_alpha_vantage_api_key() -> str:
    """Get Alpha Vantage API key from environment."""
    key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
    return key


def get_fundamentals_fetcher(
    api_key: Annotated[str, Depends(get_alpha_vantage_api_key)],
    base_path: Annotated[Path, Depends(get_data_base_path)],
) -> FundamentalsFetcher:
    """Get the fundamentals fetcher with injected dependencies."""
    return FundamentalsFetcher(
        api_key=api_key,
        base_path=base_path,
        daily_limit=500,  # Soft limit - Alpha Vantage still returns data
    )



"""Signal endpoints for news sentiment and SAC price inputs.

This package provides endpoints for extracting signals used in model training
and inference:
- News sentiment (real-time and historical)
"""

# Re-export dependencies for testing
from brain_api.routes.signals.dependencies import (
    get_data_base_path,
    get_news_fetcher,
    get_sentiment_parquet_path,
    get_sentiment_scorer,
)
from brain_api.routes.signals.endpoints import router

# Re-export models for backward compatibility
from brain_api.routes.signals.models import (
    ArticleResponse,
    HistoricalNewsSentimentRequest,
    HistoricalNewsSentimentResponse,
    NewsSignalRequest,
    NewsSignalResponse,
    SentimentDataPoint,
    SymbolSentimentResponse,
)

__all__ = [
    "ArticleResponse",
    "HistoricalNewsSentimentRequest",
    "HistoricalNewsSentimentResponse",
    # Models
    "NewsSignalRequest",
    "NewsSignalResponse",
    "SentimentDataPoint",
    "SymbolSentimentResponse",
    "get_data_base_path",
    # Dependencies
    "get_news_fetcher",
    "get_sentiment_parquet_path",
    "get_sentiment_scorer",
    "router",
]

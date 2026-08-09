"""Backward compatibility re-export.

The signals module has been split into the brain_api.routes.signals package.
This module re-exports for backward compatibility.
"""

from brain_api.routes.signals import (
    ArticleResponse,
    HistoricalNewsSentimentRequest,
    HistoricalNewsSentimentResponse,
    NewsSignalRequest,
    NewsSignalResponse,
    SentimentDataPoint,
    SymbolSentimentResponse,
    get_data_base_path,
    get_news_fetcher,
    get_sentiment_parquet_path,
    get_sentiment_scorer,
    router,
)

__all__ = [
    "ArticleResponse",
    "HistoricalNewsSentimentRequest",
    "HistoricalNewsSentimentResponse",
    "NewsSignalRequest",
    "NewsSignalResponse",
    "SentimentDataPoint",
    "SymbolSentimentResponse",
    "get_data_base_path",
    "get_news_fetcher",
    "get_sentiment_parquet_path",
    "get_sentiment_scorer",
    "router",
]

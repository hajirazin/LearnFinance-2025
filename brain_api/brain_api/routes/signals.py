"""Backward compatibility re-export.

The signals module has been split into the brain_api.routes.signals package.
This module re-exports for backward compatibility.
"""

from brain_api.routes.signals import (
    ApiStatusResponse,
    ArticleResponse,
    CurrentRatiosResponse,
    FundamentalsRequest,
    FundamentalsResponse,
    HistoricalFundamentalsRequest,
    HistoricalFundamentalsResponse,
    HistoricalNewsSentimentRequest,
    HistoricalNewsSentimentResponse,
    NewsSignalRequest,
    NewsSignalResponse,
    RatiosResponse,
    SentimentDataPoint,
    SymbolSentimentResponse,
    get_alpha_vantage_api_key,
    get_data_base_path,
    get_fundamentals_fetcher,
    get_news_fetcher,
    get_sentiment_parquet_path,
    get_sentiment_scorer,
    router,
)

__all__ = [
    "router",
    "NewsSignalRequest",
    "ArticleResponse",
    "SymbolSentimentResponse",
    "NewsSignalResponse",
    "HistoricalNewsSentimentRequest",
    "SentimentDataPoint",
    "HistoricalNewsSentimentResponse",
    "FundamentalsRequest",
    "HistoricalFundamentalsRequest",
    "RatiosResponse",
    "CurrentRatiosResponse",
    "ApiStatusResponse",
    "FundamentalsResponse",
    "HistoricalFundamentalsResponse",
    "get_news_fetcher",
    "get_sentiment_scorer",
    "get_data_base_path",
    "get_sentiment_parquet_path",
    "get_alpha_vantage_api_key",
    "get_fundamentals_fetcher",
]

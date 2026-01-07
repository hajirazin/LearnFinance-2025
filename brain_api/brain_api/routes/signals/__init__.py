"""Signal endpoints for feature extraction (news sentiment, fundamentals, etc.).

This package provides endpoints for extracting signals used in model training
and inference:
- News sentiment (real-time and historical)
- Fundamentals (current via yfinance, historical via Alpha Vantage)
"""

from brain_api.routes.signals.endpoints import router

# Re-export models for backward compatibility
from brain_api.routes.signals.models import (
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
)

# Re-export dependencies for testing
from brain_api.routes.signals.dependencies import (
    get_alpha_vantage_api_key,
    get_data_base_path,
    get_fundamentals_fetcher,
    get_news_fetcher,
    get_sentiment_parquet_path,
    get_sentiment_scorer,
)

__all__ = [
    "router",
    # Models
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
    # Dependencies
    "get_news_fetcher",
    "get_sentiment_scorer",
    "get_data_base_path",
    "get_sentiment_parquet_path",
    "get_alpha_vantage_api_key",
    "get_fundamentals_fetcher",
]



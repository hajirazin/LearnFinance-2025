"""Signal endpoints for feature extraction (news sentiment, fundamentals, etc.).

This package provides endpoints for extracting signals used in model training
and inference:
- News sentiment (real-time and historical)
- Fundamentals (current via yfinance, historical via Alpha Vantage)
"""

# Re-export dependencies for testing
from brain_api.routes.signals.dependencies import (
    get_alpha_vantage_api_key,
    get_data_base_path,
    get_fundamentals_fetcher,
    get_news_fetcher,
    get_sentiment_parquet_path,
    get_sentiment_scorer,
)
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
    RefreshFundamentalsRequest,
    RefreshFundamentalsResponse,
    SentimentDataPoint,
    SymbolSentimentResponse,
)

__all__ = [
    "ApiStatusResponse",
    "ArticleResponse",
    "CurrentRatiosResponse",
    "FundamentalsRequest",
    "FundamentalsResponse",
    "HistoricalFundamentalsRequest",
    "HistoricalFundamentalsResponse",
    "HistoricalNewsSentimentRequest",
    "HistoricalNewsSentimentResponse",
    # Models
    "NewsSignalRequest",
    "NewsSignalResponse",
    "RatiosResponse",
    "RefreshFundamentalsRequest",
    "RefreshFundamentalsResponse",
    "SentimentDataPoint",
    "SymbolSentimentResponse",
    "get_alpha_vantage_api_key",
    "get_data_base_path",
    "get_fundamentals_fetcher",
    # Dependencies
    "get_news_fetcher",
    "get_sentiment_parquet_path",
    "get_sentiment_scorer",
    "router",
]

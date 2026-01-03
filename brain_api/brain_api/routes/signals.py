"""Signal endpoints for feature extraction (news sentiment, fundamentals, etc.).

This module re-exports from signals_news.py and signals_fundamentals.py
for backward compatibility. New code should import from the specific modules.
"""

from fastapi import APIRouter

from brain_api.routes.signals_fundamentals import (
    router as fundamentals_router,
)
from brain_api.routes.signals_fundamentals import (
    # Request/Response models
    ApiStatusResponse,
    CurrentRatiosResponse,
    FundamentalsRequest,
    FundamentalsResponse,
    HistoricalFundamentalsRequest,
    HistoricalFundamentalsResponse,
    RatiosResponse,
    # Endpoints
    get_fundamentals,
    get_historical_fundamentals,
    # Dependencies
    get_alpha_vantage_api_key,
    get_fundamentals_fetcher,
)
from brain_api.routes.signals_news import (
    router as news_router,
)
from brain_api.routes.signals_news import (
    # Request/Response models
    ArticleResponse,
    HistoricalNewsSentimentRequest,
    HistoricalNewsSentimentResponse,
    NewsSignalRequest,
    NewsSignalResponse,
    SentimentDataPoint,
    SymbolSentimentResponse,
    # Endpoints
    get_historical_news_sentiment,
    get_news_sentiment,
    # Dependencies
    get_data_base_path,
    get_news_fetcher,
    get_sentiment_parquet_path,
    get_sentiment_scorer,
)

# Create a combined router for backward compatibility
router = APIRouter()

# Include sub-routers
router.include_router(news_router, tags=["signals"])
router.include_router(fundamentals_router, tags=["signals"])

__all__ = [
    # Routers
    "router",
    "news_router",
    "fundamentals_router",
    # News request/response
    "NewsSignalRequest",
    "ArticleResponse",
    "SymbolSentimentResponse",
    "NewsSignalResponse",
    "HistoricalNewsSentimentRequest",
    "SentimentDataPoint",
    "HistoricalNewsSentimentResponse",
    # Fundamentals request/response
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
    # Endpoints
    "get_news_sentiment",
    "get_historical_news_sentiment",
    "get_fundamentals",
    "get_historical_fundamentals",
]

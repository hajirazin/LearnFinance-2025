"""Signal endpoints for news sentiment and SAC price inputs.

This package provides endpoints for extracting signals used in model training
and inference:
- News sentiment (real-time and historical)
"""

from fastapi import APIRouter

# Re-export dependencies for testing
from brain_api.routes.signals.dependencies import (
    get_data_base_path,
    get_news_fetcher,
    get_sentiment_parquet_path,
    get_sentiment_scorer,
)
from brain_api.routes.signals.endpoints import router as endpoints_router

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
from brain_api.routes.signals.ppo_discovery import router as ppo_discovery_router

router = APIRouter()
router.include_router(endpoints_router)
router.include_router(ppo_discovery_router)

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

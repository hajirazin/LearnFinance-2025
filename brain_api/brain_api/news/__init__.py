"""Independent news bounded context."""

from brain_api.news.aggregation import (
    confidence_recency_weighted_mean,
    population_std,
)
from brain_api.news.errors import (
    NewsCapExceeded,
    NewsCoverageMissing,
    NewsError,
    NewsProviderError,
    SentimentScoringError,
)
from brain_api.news.models import (
    CONFIDENCE_RECENCY_TAU_HOURS,
    NEWS_PROVIDER,
    NEWS_SCHEMA_VERSION,
    NEWS_SENTIMENT_MODEL,
    NEWS_SENTIMENT_REVISION,
    NewsEvent,
    NewsWindow,
)
from brain_api.news.service import NewsService
from brain_api.news.store import NewsStore

__all__ = [
    "CONFIDENCE_RECENCY_TAU_HOURS",
    "NEWS_PROVIDER",
    "NEWS_SCHEMA_VERSION",
    "NEWS_SENTIMENT_MODEL",
    "NEWS_SENTIMENT_REVISION",
    "NewsCapExceeded",
    "NewsCoverageMissing",
    "NewsError",
    "NewsEvent",
    "NewsProviderError",
    "NewsService",
    "NewsStore",
    "NewsWindow",
    "SentimentScoringError",
    "confidence_recency_weighted_mean",
    "population_std",
]

"""News sentiment analysis module.

This module handles real-time news sentiment for inference:
- Fetches news from yfinance for specific symbols
- Scores with the unified FinBERT scorer
- Aggregates per-symbol sentiment with recency weighting
"""

# Models
from brain_api.core.news_sentiment.models import (
    Article,
    NewsSentimentResult,
    ScoredArticle,
    SymbolSentiment,
)

# Protocols
from brain_api.core.news_sentiment.protocols import NewsFetcher, SentimentScorer

# Fetcher
from brain_api.core.news_sentiment.fetcher import YFinanceNewsFetcher

# Aggregation
from brain_api.core.news_sentiment.aggregation import (
    aggregate_symbol_sentiment,
    compute_recency_weight,
)

# Persistence
from brain_api.core.news_sentiment.persistence import (
    get_features_path,
    get_raw_news_path,
    load_cached_features,
    load_cached_symbol,
    save_features,
    save_raw_news,
)

# Processor
from brain_api.core.news_sentiment.processor import (
    process_news_sentiment,
    process_symbol_news,
)

__all__ = [
    # Models
    "Article",
    "ScoredArticle",
    "SymbolSentiment",
    "NewsSentimentResult",
    # Protocols
    "NewsFetcher",
    "SentimentScorer",
    # Fetcher
    "YFinanceNewsFetcher",
    # Aggregation
    "compute_recency_weight",
    "aggregate_symbol_sentiment",
    # Persistence
    "get_raw_news_path",
    "get_features_path",
    "save_raw_news",
    "save_features",
    "load_cached_features",
    "load_cached_symbol",
    # Processor
    "process_symbol_news",
    "process_news_sentiment",
]



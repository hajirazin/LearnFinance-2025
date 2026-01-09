"""News sentiment analysis module.

This module handles real-time news sentiment for inference:
- Fetches news from yfinance for specific symbols
- Scores with the unified FinBERT scorer
- Aggregates per-symbol sentiment with recency weighting
"""

# Models
# Aggregation
from brain_api.core.news_sentiment.aggregation import (
    aggregate_symbol_sentiment,
    compute_recency_weight,
)

# Fetcher
from brain_api.core.news_sentiment.fetcher import YFinanceNewsFetcher
from brain_api.core.news_sentiment.models import (
    Article,
    NewsSentimentResult,
    ScoredArticle,
    SymbolSentiment,
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

# Protocols
from brain_api.core.news_sentiment.protocols import NewsFetcher, SentimentScorer

__all__ = [
    # Models
    "Article",
    # Protocols
    "NewsFetcher",
    "NewsSentimentResult",
    "ScoredArticle",
    "SentimentScorer",
    "SymbolSentiment",
    # Fetcher
    "YFinanceNewsFetcher",
    "aggregate_symbol_sentiment",
    # Aggregation
    "compute_recency_weight",
    "get_features_path",
    # Persistence
    "get_raw_news_path",
    "load_cached_features",
    "load_cached_symbol",
    "process_news_sentiment",
    # Processor
    "process_symbol_news",
    "save_features",
    "save_raw_news",
]

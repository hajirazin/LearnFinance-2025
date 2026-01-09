"""Backward compatibility re-export.

The news sentiment module has been split into the brain_api.core.news_sentiment package.
This module re-exports for backward compatibility.
"""

from brain_api.core.news_sentiment import (
    Article,
    NewsFetcher,
    NewsSentimentResult,
    ScoredArticle,
    SentimentScorer,
    SymbolSentiment,
    YFinanceNewsFetcher,
    aggregate_symbol_sentiment,
    compute_recency_weight,
    get_features_path,
    get_raw_news_path,
    load_cached_features,
    load_cached_symbol,
    process_news_sentiment,
    process_symbol_news,
    save_features,
    save_raw_news,
)

__all__ = [
    "Article",
    "NewsFetcher",
    "NewsSentimentResult",
    "ScoredArticle",
    "SentimentScorer",
    "SymbolSentiment",
    "YFinanceNewsFetcher",
    "aggregate_symbol_sentiment",
    "compute_recency_weight",
    "get_features_path",
    "get_raw_news_path",
    "load_cached_features",
    "load_cached_symbol",
    "process_news_sentiment",
    "process_symbol_news",
    "save_features",
    "save_raw_news",
]

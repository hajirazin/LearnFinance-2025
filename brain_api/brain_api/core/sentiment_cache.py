"""Backward compatibility re-export.

The sentiment cache has been moved to brain_api.storage.sentiment_cache.
This module re-exports for backward compatibility.
"""

from brain_api.storage.sentiment_cache import CacheStats, SentimentCache

__all__ = ["CacheStats", "SentimentCache"]

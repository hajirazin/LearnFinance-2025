"""Errors for the news bounded context."""

from __future__ import annotations


class NewsError(Exception):
    """Base error for the news domain."""


class NewsProviderError(NewsError):
    """Alpaca/Benzinga fetch failed or was incomplete."""


class SentimentScoringError(NewsError):
    """FinBERT could not score required text."""


class NewsWindowNotClosed(NewsError):
    """Materialize refused a window whose end is still in the future."""


class NewsCapExceeded(NewsError):
    """Unique-article cap exceeded for a symbol window or request."""


class NewsCoverageMissing(NewsError):
    """Query asked for a window with no complete coverage row."""


class NewsStoreConflict(NewsError):
    """DuckDB writer conflict after retries."""


class RepeatedPageTokenError(NewsProviderError):
    """Pagination returned the same page token twice."""

"""News provider protocol."""

from __future__ import annotations

from typing import Protocol

from brain_api.news.models import NewsWindow, ProviderArticle


class NewsProvider(Protocol):
    """Fetch articles for one symbol and window. Pagination stays inside."""

    def fetch_window(
        self, symbol: str, window: NewsWindow
    ) -> tuple[list[ProviderArticle], int]:
        """Return ``(articles, page_count)``. Must exhaust pages or raise."""

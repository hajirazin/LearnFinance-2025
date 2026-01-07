"""Protocol definitions for dependency injection."""

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from brain_api.core.finbert import SentimentScore
    from brain_api.core.news_sentiment.models import Article


class NewsFetcher(Protocol):
    """Protocol for fetching news articles."""

    def fetch(self, symbol: str, max_articles: int) -> list["Article"]:
        """Fetch news articles for a symbol."""
        ...


class SentimentScorer(Protocol):
    """Protocol for scoring article sentiment."""

    def score(self, text: str) -> "SentimentScore":
        """Score the sentiment of a text."""
        ...

    def score_batch(self, texts: list[str]) -> list["SentimentScore"]:
        """Score sentiment for a batch of texts."""
        ...



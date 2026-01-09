"""Data models for news sentiment module."""

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from brain_api.core.finbert import SentimentScore


@dataclass
class Article:
    """A single news article from yfinance."""

    title: str
    publisher: str
    link: str
    published: datetime | None  # May not be available
    summary: str | None  # yfinance sometimes includes a summary/snippet

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "title": self.title,
            "publisher": self.publisher,
            "link": self.link,
            "published": self.published.isoformat() if self.published else None,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Article":
        """Create from dictionary."""
        published = None
        if data.get("published"):
            published = datetime.fromisoformat(data["published"])
        return cls(
            title=data["title"],
            publisher=data["publisher"],
            link=data["link"],
            published=published,
            summary=data.get("summary"),
        )


@dataclass
class ScoredArticle:
    """An article with its FinBERT sentiment score."""

    article: Article
    sentiment: "SentimentScore"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "article": self.article.to_dict(),
            "sentiment": self.sentiment.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScoredArticle":
        """Create from dictionary."""
        from brain_api.core.finbert import SentimentScore

        return cls(
            article=Article.from_dict(data["article"]),
            sentiment=SentimentScore.from_dict(data["sentiment"]),
        )


@dataclass
class SymbolSentiment:
    """Aggregated sentiment result for a single symbol."""

    symbol: str
    article_count_fetched: int
    article_count_used: int  # After filtering bad data
    sentiment_score: float  # Recency-weighted average of article_score
    insufficient_news: bool  # True if < 3 usable articles
    scored_articles: list[ScoredArticle]  # All scored articles

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "article_count_fetched": self.article_count_fetched,
            "article_count_used": self.article_count_used,
            "sentiment_score": self.sentiment_score,
            "insufficient_news": self.insufficient_news,
            "scored_articles": [a.to_dict() for a in self.scored_articles],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SymbolSentiment":
        """Create from dictionary."""
        return cls(
            symbol=data["symbol"],
            article_count_fetched=data["article_count_fetched"],
            article_count_used=data["article_count_used"],
            sentiment_score=data["sentiment_score"],
            insufficient_news=data["insufficient_news"],
            scored_articles=[
                ScoredArticle.from_dict(a) for a in data["scored_articles"]
            ],
        )


@dataclass
class NewsSentimentResult:
    """Full result of news sentiment processing."""

    run_id: str
    attempt: int
    as_of_date: str
    per_symbol: list[SymbolSentiment]
    from_cache: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "run_id": self.run_id,
            "attempt": self.attempt,
            "as_of_date": self.as_of_date,
            "per_symbol": [s.to_dict() for s in self.per_symbol],
            "from_cache": self.from_cache,
        }

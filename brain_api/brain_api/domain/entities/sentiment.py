"""Sentiment-related domain entities."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any


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
class FinBERTResult:
    """FinBERT sentiment classification result for an article."""

    label: str  # "positive", "negative", or "neutral"
    p_pos: float
    p_neg: float
    p_neu: float
    article_score: float  # p_pos - p_neg, range [-1, 1]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "label": self.label,
            "p_pos": self.p_pos,
            "p_neg": self.p_neg,
            "p_neu": self.p_neu,
            "article_score": self.article_score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FinBERTResult":
        """Create from dictionary."""
        return cls(
            label=data["label"],
            p_pos=data["p_pos"],
            p_neg=data["p_neg"],
            p_neu=data["p_neu"],
            article_score=data["article_score"],
        )


@dataclass
class ScoredArticle:
    """An article with its FinBERT sentiment score."""

    article: Article
    finbert: FinBERTResult

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "article": self.article.to_dict(),
            "finbert": self.finbert.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScoredArticle":
        """Create from dictionary."""
        return cls(
            article=Article.from_dict(data["article"]),
            finbert=FinBERTResult.from_dict(data["finbert"]),
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


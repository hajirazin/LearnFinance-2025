"""News sentiment analysis using yfinance news + FinBERT scoring."""

import json
import math
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Protocol

import yfinance as yf

from shared.ml.finbert import FinBERTResult as SharedFinBERTResult
from shared.ml.finbert import FinBERTScorer as SharedFinBERTScorer

# ============================================================================
# Data models
# ============================================================================


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


# ============================================================================
# Protocols for dependency injection
# ============================================================================


class NewsFetcher(Protocol):
    """Protocol for fetching news articles."""

    def fetch(self, symbol: str, max_articles: int) -> list[Article]:
        """Fetch news articles for a symbol."""
        ...


class SentimentScorer(Protocol):
    """Protocol for scoring article sentiment."""

    def score(self, text: str) -> FinBERTResult:
        """Score the sentiment of a text."""
        ...

    def score_batch(self, texts: list[str]) -> list[FinBERTResult]:
        """Score sentiment for a batch of texts."""
        ...


# ============================================================================
# yfinance news fetcher implementation
# ============================================================================


class YFinanceNewsFetcher:
    """Fetch news articles from yfinance."""

    def fetch(self, symbol: str, max_articles: int) -> list[Article]:
        """Fetch news articles for a symbol using yfinance.

        Args:
            symbol: Stock ticker symbol
            max_articles: Maximum number of articles to fetch

        Returns:
            List of Article objects
        """
        try:
            ticker = yf.Ticker(symbol)
            news_data = ticker.news or []
        except Exception:
            return []

        articles = []
        for item in news_data[:max_articles]:
            # yfinance news structure changed: data is nested in 'content'
            content = item.get("content", item)  # Fallback to item if no content key

            # Parse published timestamp (now ISO string in pubDate)
            published = None
            pub_date_str = content.get("pubDate")
            if pub_date_str:
                with suppress(ValueError, TypeError):
                    # Parse ISO format: "2025-12-29T21:55:58Z"
                    published = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
            # Fallback to old format (Unix timestamp)
            elif "providerPublishTime" in item:
                with suppress(ValueError, OSError):
                    published = datetime.fromtimestamp(item["providerPublishTime"], tz=UTC)

            # Extract article data from new structure
            # Publisher is now in content.provider.displayName
            provider = content.get("provider", {})
            publisher = provider.get("displayName", "") if isinstance(provider, dict) else ""

            # Link is now in content.canonicalUrl.url or content.clickThroughUrl.url
            link = ""
            canonical_url = content.get("canonicalUrl", {})
            if isinstance(canonical_url, dict) and canonical_url.get("url"):
                link = canonical_url["url"]
            else:
                click_url = content.get("clickThroughUrl", {})
                if isinstance(click_url, dict) and click_url.get("url"):
                    link = click_url["url"]

            # Title and summary
            title = content.get("title", "")
            summary = content.get("summary")

            article = Article(
                title=title,
                publisher=publisher,
                link=link,
                published=published,
                summary=summary,
            )

            # Only include articles with a title
            if article.title:
                articles.append(article)

        return articles


# ============================================================================
# FinBERT sentiment scorer implementation
# ============================================================================


def _shared_to_local_result(shared: SharedFinBERTResult) -> FinBERTResult:
    """Convert shared FinBERTResult to local format."""
    return FinBERTResult(
        label=shared.label,
        p_pos=shared.p_pos,
        p_neg=shared.p_neg,
        p_neu=shared.p_neu,
        article_score=shared.score,
    )


class FinBERTScorer:
    """Score article sentiment using FinBERT (ProsusAI/finbert).

    This is a wrapper around the shared FinBERTScorer that converts
    results to the local FinBERTResult format for backward compatibility.
    """

    def __init__(self):
        """Initialize with shared scorer instance."""
        self._shared_scorer = SharedFinBERTScorer()

    def score(self, text: str) -> FinBERTResult:
        """Score the sentiment of a single text.

        Args:
            text: Text to analyze (typically article title or title + summary)

        Returns:
            FinBERTResult with label, probabilities, and article_score
        """
        shared_result = self._shared_scorer.score(text)
        return _shared_to_local_result(shared_result)

    def score_batch(self, texts: list[str]) -> list[FinBERTResult]:
        """Score sentiment for a batch of texts.

        Args:
            texts: List of texts to analyze

        Returns:
            List of FinBERTResult objects
        """
        shared_results = self._shared_scorer.score_batch(texts)
        return [_shared_to_local_result(r) for r in shared_results]


# ============================================================================
# Aggregation logic
# ============================================================================


def compute_recency_weight(
    published: datetime | None,
    as_of: datetime,
    tau_days: float = 7.0,
) -> float:
    """Compute recency weight for an article.

    Uses exponential decay: weight = exp(-age_days / tau_days)

    Args:
        published: Article publication time (if unknown, returns 0.5)
        as_of: Reference time for computing age
        tau_days: Decay constant in days (default 7 = one week half-life)

    Returns:
        Weight in range (0, 1]
    """
    if published is None:
        # Unknown date: use a middle-ground weight
        return 0.5

    # Compute age in days
    if published.tzinfo is None:
        published = published.replace(tzinfo=UTC)
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=UTC)

    age_seconds = (as_of - published).total_seconds()
    age_days = max(0, age_seconds / 86400)  # Clamp negative ages to 0

    return math.exp(-age_days / tau_days)


def aggregate_symbol_sentiment(
    symbol: str,
    scored_articles: list[ScoredArticle],
    as_of: datetime,
    tau_days: float = 7.0,
    min_articles: int = 3,
) -> SymbolSentiment:
    """Compute aggregated sentiment for a symbol from scored articles.

    Uses recency-weighted average of article scores.

    Args:
        symbol: Stock ticker symbol
        scored_articles: List of articles with FinBERT scores
        as_of: Reference time for recency weighting
        tau_days: Decay constant for recency weighting
        min_articles: Minimum articles needed for confidence

    Returns:
        SymbolSentiment with aggregated score and metadata
    """
    if not scored_articles:
        return SymbolSentiment(
            symbol=symbol,
            article_count_fetched=0,
            article_count_used=0,
            sentiment_score=0.0,
            insufficient_news=True,
            scored_articles=[],
        )

    # Compute weighted average
    total_weight = 0.0
    weighted_sum = 0.0

    for sa in scored_articles:
        weight = compute_recency_weight(sa.article.published, as_of, tau_days)
        weighted_sum += weight * sa.finbert.article_score
        total_weight += weight

    sentiment_score = weighted_sum / total_weight if total_weight > 0 else 0.0

    return SymbolSentiment(
        symbol=symbol,
        article_count_fetched=len(scored_articles),
        article_count_used=len(scored_articles),
        sentiment_score=round(sentiment_score, 4),
        insufficient_news=len(scored_articles) < min_articles,
        scored_articles=scored_articles,
    )


# ============================================================================
# Main processing function
# ============================================================================


def process_symbol_news(
    symbol: str,
    fetcher: NewsFetcher,
    scorer: SentimentScorer,
    max_articles: int,
    as_of: datetime,
    tau_days: float = 7.0,
) -> SymbolSentiment:
    """Fetch and score news for a single symbol.

    Args:
        symbol: Stock ticker symbol
        fetcher: NewsFetcher implementation
        scorer: SentimentScorer implementation
        max_articles: Maximum articles to fetch
        as_of: Reference time for recency weighting
        tau_days: Decay constant for recency weighting

    Returns:
        SymbolSentiment with scored articles and aggregated score
    """
    # Fetch articles
    articles = fetcher.fetch(symbol, max_articles)

    if not articles:
        return SymbolSentiment(
            symbol=symbol,
            article_count_fetched=0,
            article_count_used=0,
            sentiment_score=0.0,
            insufficient_news=True,
            scored_articles=[],
        )

    # Prepare texts for scoring (use title + summary if available)
    texts = []
    for article in articles:
        if article.summary:
            text = f"{article.title}. {article.summary}"
        else:
            text = article.title
        texts.append(text)

    # Score all articles in batch
    finbert_results = scorer.score_batch(texts)

    # Combine articles with scores
    scored_articles = [
        ScoredArticle(article=article, finbert=finbert)
        for article, finbert in zip(articles, finbert_results, strict=True)
    ]

    # Aggregate
    return aggregate_symbol_sentiment(
        symbol=symbol,
        scored_articles=scored_articles,
        as_of=as_of,
        tau_days=tau_days,
    )


# ============================================================================
# Persistence helpers
# ============================================================================


def get_raw_news_path(base_path: Path, run_id: str, attempt: int, symbol: str) -> Path:
    """Get path for raw news JSON file.

    Args:
        base_path: Base data directory
        run_id: Run identifier (e.g., "paper:2025-12-30")
        attempt: Attempt number
        symbol: Stock ticker symbol

    Returns:
        Path to the raw news JSON file
    """
    # Sanitize run_id for filesystem (replace colons)
    safe_run_id = run_id.replace(":", "_")
    return base_path / "raw" / safe_run_id / str(attempt) / "news" / f"{symbol}.json"


def get_features_path(base_path: Path, run_id: str, attempt: int) -> Path:
    """Get path for aggregated features JSON file.

    Args:
        base_path: Base data directory
        run_id: Run identifier
        attempt: Attempt number

    Returns:
        Path to the news_sentiment.json file
    """
    safe_run_id = run_id.replace(":", "_")
    return base_path / "features" / safe_run_id / str(attempt) / "news_sentiment.json"


def save_raw_news(
    base_path: Path,
    run_id: str,
    attempt: int,
    symbol: str,
    sentiment: SymbolSentiment,
) -> Path:
    """Save raw news data for a symbol.

    Args:
        base_path: Base data directory
        run_id: Run identifier
        attempt: Attempt number
        symbol: Stock ticker symbol
        sentiment: SymbolSentiment to save

    Returns:
        Path where data was saved
    """
    path = get_raw_news_path(base_path, run_id, attempt, symbol)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(sentiment.to_dict(), f, indent=2)

    return path


def save_features(
    base_path: Path,
    run_id: str,
    attempt: int,
    as_of_date: str,
    sentiments: list[SymbolSentiment],
) -> Path:
    """Save aggregated sentiment features.

    Args:
        base_path: Base data directory
        run_id: Run identifier
        attempt: Attempt number
        as_of_date: Reference date (ISO format)
        sentiments: List of SymbolSentiment results

    Returns:
        Path where data was saved
    """
    path = get_features_path(base_path, run_id, attempt)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "run_id": run_id,
        "attempt": attempt,
        "as_of_date": as_of_date,
        "timestamp": datetime.now(UTC).isoformat(),
        "per_symbol": [s.to_dict() for s in sentiments],
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    return path


def load_cached_features(
    base_path: Path,
    run_id: str,
    attempt: int,
) -> dict[str, Any] | None:
    """Load cached features if they exist.

    Args:
        base_path: Base data directory
        run_id: Run identifier
        attempt: Attempt number

    Returns:
        Cached features dict if exists, None otherwise
    """
    path = get_features_path(base_path, run_id, attempt)
    if not path.exists():
        return None

    with open(path) as f:
        return json.load(f)


def load_cached_symbol(
    base_path: Path,
    run_id: str,
    attempt: int,
    symbol: str,
) -> SymbolSentiment | None:
    """Load cached sentiment for a single symbol if it exists.

    Args:
        base_path: Base data directory
        run_id: Run identifier
        attempt: Attempt number
        symbol: Stock ticker symbol

    Returns:
        SymbolSentiment if cached, None otherwise
    """
    path = get_raw_news_path(base_path, run_id, attempt, symbol)
    if not path.exists():
        return None

    with open(path) as f:
        data = json.load(f)
        return SymbolSentiment.from_dict(data)


# ============================================================================
# Full processing pipeline
# ============================================================================


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


def process_news_sentiment(
    symbols: list[str],
    fetcher: NewsFetcher,
    scorer: SentimentScorer,
    as_of_date: date,
    max_articles_per_symbol: int = 10,
    run_id: str | None = None,
    attempt: int = 1,
    base_path: Path | None = None,
    tau_days: float = 7.0,
) -> NewsSentimentResult:
    """Process news sentiment for multiple symbols.

    Fetches news, scores with FinBERT, aggregates, and persists results.
    If results already exist for run_id+attempt, returns cached data.

    Args:
        symbols: List of stock ticker symbols
        fetcher: NewsFetcher implementation
        scorer: SentimentScorer implementation
        as_of_date: Reference date for the run
        max_articles_per_symbol: Max articles to fetch per symbol
        run_id: Run identifier (defaults to paper:<as_of_date>)
        attempt: Attempt number (defaults to 1)
        base_path: Base data directory (defaults to data/)
        tau_days: Decay constant for recency weighting

    Returns:
        NewsSentimentResult with all sentiment data
    """
    # Set defaults
    if run_id is None:
        run_id = f"paper:{as_of_date.isoformat()}"
    if base_path is None:
        base_path = Path("data")

    # Check for cached result
    cached = load_cached_features(base_path, run_id, attempt)
    if cached is not None:
        # Return cached data
        return NewsSentimentResult(
            run_id=cached["run_id"],
            attempt=cached["attempt"],
            as_of_date=cached["as_of_date"],
            per_symbol=[
                SymbolSentiment.from_dict(s) for s in cached["per_symbol"]
            ],
            from_cache=True,
        )

    # Process each symbol
    as_of_datetime = datetime.combine(as_of_date, datetime.min.time(), tzinfo=UTC)
    sentiments: list[SymbolSentiment] = []

    for symbol in symbols:
        # Check for cached symbol data
        cached_symbol = load_cached_symbol(base_path, run_id, attempt, symbol)
        if cached_symbol is not None:
            sentiments.append(cached_symbol)
            continue

        # Process fresh
        sentiment = process_symbol_news(
            symbol=symbol,
            fetcher=fetcher,
            scorer=scorer,
            max_articles=max_articles_per_symbol,
            as_of=as_of_datetime,
            tau_days=tau_days,
        )
        sentiments.append(sentiment)

        # Save raw news immediately
        save_raw_news(base_path, run_id, attempt, symbol, sentiment)

    # Save aggregated features
    save_features(base_path, run_id, attempt, as_of_date.isoformat(), sentiments)

    return NewsSentimentResult(
        run_id=run_id,
        attempt=attempt,
        as_of_date=as_of_date.isoformat(),
        per_symbol=sentiments,
        from_cache=False,
    )


"""Confidence-weighted daily sentiment aggregation with bounded filtering."""

from collections import defaultdict
from dataclasses import dataclass

from news_sentiment_etl.core.dataset import NewsArticle
from news_sentiment_etl.core.sentiment import SentimentScore


@dataclass
class DailySentiment:
    """Aggregated daily sentiment for a symbol.

    Attributes:
        date: Trading date (YYYY-MM-DD)
        symbol: Stock ticker
        sentiment_score: Confidence-weighted average score [-1, 1]
        article_count: Number of articles used after filtering
        avg_confidence: Average FinBERT confidence
        p_pos_avg: Average positive probability
        p_neg_avg: Average negative probability
        total_articles: Total articles before bounded filtering
    """

    date: str
    symbol: str
    sentiment_score: float
    article_count: int
    avg_confidence: float
    p_pos_avg: float
    p_neg_avg: float
    total_articles: int

    def to_dict(self) -> dict:
        """Convert to dictionary for output."""
        return {
            "date": self.date,
            "symbol": self.symbol,
            "sentiment_score": self.sentiment_score,
            "article_count": self.article_count,
            "avg_confidence": self.avg_confidence,
            "p_pos_avg": self.p_pos_avg,
            "p_neg_avg": self.p_neg_avg,
            "total_articles": self.total_articles,
        }


@dataclass
class ArticleWithScore:
    """An article with its sentiment score."""

    article: NewsArticle
    score: SentimentScore


class SentimentAggregator:
    """Aggregates article sentiments into daily scores per symbol.

    Uses confidence-weighted averaging with bounded filtering:
    1. Filter out articles where |p_pos - p_neg| < threshold (too ambiguous)
    2. Weight remaining by confidence = max(p_pos, p_neg, p_neu)
    3. daily_sentiment = sum(confidence * score) / sum(confidence)
    """

    def __init__(self, threshold: float = 0.1):
        """Initialize aggregator.

        Args:
            threshold: Minimum |p_pos - p_neg| to include an article
        """
        self.threshold = threshold
        # Map of (date, symbol) -> list of ArticleWithScore
        self._articles: dict[tuple[str, str], list[ArticleWithScore]] = defaultdict(
            list
        )

    def add(self, article: NewsArticle, score: SentimentScore) -> None:
        """Add an article with its score.

        The article is added for each symbol it mentions.

        Args:
            article: The news article
            score: FinBERT sentiment score
        """
        for symbol in article.symbols:
            key = (article.date, symbol)
            self._articles[key].append(ArticleWithScore(article, score))

    def add_batch(
        self, articles: list[NewsArticle], scores: list[SentimentScore]
    ) -> None:
        """Add a batch of articles with their scores.

        Args:
            articles: List of news articles
            scores: Corresponding FinBERT scores
        """
        for article, score in zip(articles, scores, strict=True):
            self.add(article, score)

    def aggregate(self) -> list[DailySentiment]:
        """Compute aggregated daily sentiments.

        Returns:
            List of DailySentiment objects, one per (date, symbol) pair
        """
        results = []

        for (date, symbol), articles_with_scores in self._articles.items():
            # Separate into those that pass threshold and those that don't
            passed = [
                aws for aws in articles_with_scores
                if aws.score.passes_threshold(self.threshold)
            ]
            total_articles = len(articles_with_scores)

            if not passed:
                # No articles passed the threshold - skip this day/symbol
                continue

            # Compute confidence-weighted average
            total_weight = 0.0
            weighted_sum = 0.0
            sum_confidence = 0.0
            sum_p_pos = 0.0
            sum_p_neg = 0.0

            for aws in passed:
                weight = aws.score.confidence
                weighted_sum += weight * aws.score.score
                total_weight += weight
                sum_confidence += aws.score.confidence
                sum_p_pos += aws.score.p_pos
                sum_p_neg += aws.score.p_neg

            article_count = len(passed)
            sentiment_score = weighted_sum / total_weight if total_weight > 0 else 0.0

            results.append(
                DailySentiment(
                    date=date,
                    symbol=symbol,
                    sentiment_score=round(sentiment_score, 4),
                    article_count=article_count,
                    avg_confidence=round(sum_confidence / article_count, 4),
                    p_pos_avg=round(sum_p_pos / article_count, 4),
                    p_neg_avg=round(sum_p_neg / article_count, 4),
                    total_articles=total_articles,
                )
            )

        # Sort by date, then symbol
        results.sort(key=lambda x: (x.date, x.symbol))
        return results

    def clear(self) -> None:
        """Clear all accumulated articles."""
        self._articles.clear()

    @property
    def pending_count(self) -> int:
        """Number of (date, symbol) pairs waiting to be aggregated."""
        return len(self._articles)

    @property
    def article_count(self) -> int:
        """Total number of articles added."""
        return sum(len(v) for v in self._articles.values())


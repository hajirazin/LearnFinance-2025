"""Daily sentiment dataclass for aggregation output.

Moved from news_sentiment_etl/core/aggregation.py.
"""

from dataclasses import dataclass


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

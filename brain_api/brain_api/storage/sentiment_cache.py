"""SQLite-based sentiment cache for FinBERT scores.

Moved from news_sentiment_etl/core/cache.py and updated to use
the unified SentimentScore from finbert.py.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

if TYPE_CHECKING:
    from brain_api.core.finbert import SentimentScore

console = Console()


@dataclass
class CacheStats:
    """Statistics about cache usage."""

    total_entries: int
    hits: int = 0
    misses: int = 0

    @property
    def hit_rate(self) -> float:
        """Return cache hit rate as percentage."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return (self.hits / total) * 100


class SentimentCache:
    """SQLite-based cache for FinBERT sentiment scores.

    Stores sentiment scores keyed by article text hash.
    On reruns, cached articles are retrieved instantly without
    calling FinBERT again.
    """

    def __init__(self, cache_dir: Path):
        """Initialize the cache.

        Args:
            cache_dir: Directory to store the cache database
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "sentiment_cache.db"
        self._conn: sqlite3.Connection | None = None
        self._stats = CacheStats(total_entries=0)

    def _ensure_connected(self) -> sqlite3.Connection:
        """Ensure database connection and schema exist."""
        if self._conn is not None:
            return self._conn

        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_cache (
                article_hash TEXT PRIMARY KEY,
                p_pos REAL NOT NULL,
                p_neg REAL NOT NULL,
                p_neu REAL NOT NULL,
                score REAL NOT NULL,
                confidence REAL NOT NULL,
                label TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_created
            ON sentiment_cache(created_at)
        """)
        # Article-symbol associations for aggregation (replaces in-memory dict)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS article_symbols (
                article_hash TEXT NOT NULL,
                date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                PRIMARY KEY (article_hash, date, symbol)
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_date_symbol
            ON article_symbols(date, symbol)
        """)
        self._conn.commit()

        # Get initial count
        cursor = self._conn.execute("SELECT COUNT(*) FROM sentiment_cache")
        self._stats.total_entries = cursor.fetchone()[0]

        return self._conn

    def get(self, article_hash: str) -> SentimentScore | None:
        """Get cached sentiment score by article hash.

        Args:
            article_hash: MD5 hash of article text

        Returns:
            SentimentScore if found, None otherwise
        """
        # Import here to avoid circular import
        from brain_api.core.finbert import SentimentScore

        conn = self._ensure_connected()
        cursor = conn.execute(
            """
            SELECT p_pos, p_neg, p_neu, score, confidence, label
            FROM sentiment_cache
            WHERE article_hash = ?
            """,
            (article_hash,),
        )
        row = cursor.fetchone()

        if row is None:
            self._stats.misses += 1
            return None

        self._stats.hits += 1
        return SentimentScore(
            p_pos=row[0],
            p_neg=row[1],
            p_neu=row[2],
            score=row[3],
            confidence=row[4],
            label=row[5],
        )

    def put(self, article_hash: str, score: SentimentScore) -> None:
        """Store sentiment score in cache.

        Args:
            article_hash: MD5 hash of article text
            score: FinBERT sentiment score to cache
        """
        conn = self._ensure_connected()
        now = datetime.now(UTC).isoformat()
        conn.execute(
            """
            INSERT OR REPLACE INTO sentiment_cache
            (article_hash, p_pos, p_neg, p_neu, score, confidence, label, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                article_hash,
                score.p_pos,
                score.p_neg,
                score.p_neu,
                score.score,
                score.confidence,
                score.label,
                now,
            ),
        )
        conn.commit()
        self._stats.total_entries += 1

    def get_batch(self, article_hashes: list[str]) -> dict[str, SentimentScore | None]:
        """Batch lookup of cached sentiment scores.

        Args:
            article_hashes: List of MD5 hashes to look up

        Returns:
            Dict mapping hash -> SentimentScore (or None if not found)
        """
        # Import here to avoid circular import
        from brain_api.core.finbert import SentimentScore

        if not article_hashes:
            return {}

        conn = self._ensure_connected()

        # Use IN clause for batch lookup
        placeholders = ",".join("?" * len(article_hashes))
        cursor = conn.execute(
            f"""
            SELECT article_hash, p_pos, p_neg, p_neu, score, confidence, label
            FROM sentiment_cache
            WHERE article_hash IN ({placeholders})
            """,
            article_hashes,
        )

        results: dict[str, SentimentScore | None] = dict.fromkeys(article_hashes)
        for row in cursor:
            article_hash = row[0]
            results[article_hash] = SentimentScore(
                p_pos=row[1],
                p_neg=row[2],
                p_neu=row[3],
                score=row[4],
                confidence=row[5],
                label=row[6],
            )
            self._stats.hits += 1

        # Count misses
        for h in article_hashes:
            if results[h] is None:
                self._stats.misses += 1

        return results

    def put_batch(self, hash_score_pairs: list[tuple[str, SentimentScore]]) -> None:
        """Batch insert sentiment scores into cache.

        Args:
            hash_score_pairs: List of (article_hash, SentimentScore) tuples
        """
        if not hash_score_pairs:
            return

        conn = self._ensure_connected()
        now = datetime.now(UTC).isoformat()

        conn.executemany(
            """
            INSERT OR REPLACE INTO sentiment_cache
            (article_hash, p_pos, p_neg, p_neu, score, confidence, label, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    article_hash,
                    score.p_pos,
                    score.p_neg,
                    score.p_neu,
                    score.score,
                    score.confidence,
                    score.label,
                    now,
                )
                for article_hash, score in hash_score_pairs
            ],
        )
        conn.commit()
        self._stats.total_entries += len(hash_score_pairs)

    def store_article_symbols(
        self, article_hash: str, date: str, symbols: list[str]
    ) -> None:
        """Store date/symbol associations for an article.

        Uses INSERT OR IGNORE to skip if (hash, date, symbol) already exists.

        Args:
            article_hash: MD5 hash of article text
            date: Article date in YYYY-MM-DD format
            symbols: List of stock symbols mentioned in article
        """
        if not symbols:
            return

        conn = self._ensure_connected()
        conn.executemany(
            """
            INSERT OR IGNORE INTO article_symbols (article_hash, date, symbol)
            VALUES (?, ?, ?)
            """,
            [(article_hash, date, symbol) for symbol in symbols],
        )
        conn.commit()

    @property
    def stats(self) -> CacheStats:
        """Return cache statistics."""
        return self._stats

    @property
    def article_symbols_count(self) -> int:
        """Get count of article-symbol associations."""
        conn = self._ensure_connected()
        cursor = conn.execute("SELECT COUNT(*) FROM article_symbols")
        return cursor.fetchone()[0]

    def get_all_cached_hashes(self) -> set[str]:
        """Get all article hashes that have been scored.

        Used for pre-filtering in DuckDB to skip already-cached articles.

        Returns:
            Set of article hash strings
        """
        conn = self._ensure_connected()
        cursor = conn.execute("SELECT article_hash FROM sentiment_cache")
        return {row[0] for row in cursor.fetchall()}

    def aggregate_daily_sentiment(self, threshold: float = 0.1) -> list:
        """Aggregate daily sentiment per symbol using SQL.

        Replaces the in-memory SentimentAggregator with a single SQL query.

        Args:
            threshold: Minimum |p_pos - p_neg| to include article

        Returns:
            List of DailySentiment objects
        """
        # Import here to avoid circular import
        from brain_api.etl.aggregation import DailySentiment

        conn = self._ensure_connected()
        cursor = conn.execute(
            """
            SELECT
                a.date,
                a.symbol,
                SUM(c.confidence * c.score) / SUM(c.confidence) as sentiment_score,
                COUNT(*) as article_count,
                AVG(c.confidence) as avg_confidence,
                AVG(c.p_pos) as p_pos_avg,
                AVG(c.p_neg) as p_neg_avg,
                COUNT(*) as total_articles
            FROM article_symbols a
            JOIN sentiment_cache c ON a.article_hash = c.article_hash
            WHERE ABS(c.p_pos - c.p_neg) >= ?
            GROUP BY a.date, a.symbol
            ORDER BY a.date, a.symbol
            """,
            (threshold,),
        )

        results = []
        for row in cursor:
            results.append(
                DailySentiment(
                    date=row[0],
                    symbol=row[1],
                    sentiment_score=round(row[2], 4) if row[2] else 0.0,
                    article_count=row[3],
                    avg_confidence=round(row[4], 4) if row[4] else 0.0,
                    p_pos_avg=round(row[5], 4) if row[5] else 0.0,
                    p_neg_avg=round(row[6], 4) if row[6] else 0.0,
                    total_articles=row[7],
                )
            )

        return results

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def log_status(self) -> None:
        """Log cache status to console."""
        self._ensure_connected()
        symbol_count = self.article_symbols_count
        console.print("[bold blue]🗄️ Loading sentiment cache...[/]")
        console.print(f"  Cache location: [cyan]{self.db_path}[/]")
        console.print(f"  Cached scores: [green]{self._stats.total_entries:,}[/]")
        console.print(f"  Article-symbol pairs: [green]{symbol_count:,}[/]")
        console.print()

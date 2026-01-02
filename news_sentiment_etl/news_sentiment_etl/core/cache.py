"""SQLite-based sentiment cache for FinBERT scores."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

if TYPE_CHECKING:
    from news_sentiment_etl.core.sentiment import SentimentScore

console = Console()


def compute_article_hash(text: str) -> str:
    """Compute MD5 hash of article text for cache key.

    Args:
        text: Article text

    Returns:
        16-character hex hash
    """
    return hashlib.md5(text.encode()).hexdigest()[:16]


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
        from news_sentiment_etl.core.sentiment import SentimentScore

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
        from news_sentiment_etl.core.sentiment import SentimentScore

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

        results: dict[str, SentimentScore | None] = {h: None for h in article_hashes}
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

    def put_batch(
        self, hash_score_pairs: list[tuple[str, SentimentScore]]
    ) -> None:
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

    @property
    def stats(self) -> CacheStats:
        """Return cache statistics."""
        return self._stats

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def log_status(self) -> None:
        """Log cache status to console."""
        self._ensure_connected()
        console.print("[bold blue]🗄️ Loading sentiment cache...[/]")
        console.print(f"  Cache location: [cyan]{self.db_path}[/]")
        console.print(f"  Cached scores: [green]{self._stats.total_entries:,}[/]")
        console.print()


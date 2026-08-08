"""SQLite index for tracking fetched fundamental data."""

from __future__ import annotations

import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path

from brain_api.core.fundamentals.models import FetchRecord


class FundamentalsIndex:
    """SQLite index for tracking fetched fundamental data.

    This doesn't store the actual data - just metadata about what's been fetched.
    """

    def __init__(self, cache_dir: Path):
        """Initialize the index.

        Args:
            cache_dir: Directory to store the index database
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "fundamentals.db"
        self._conn: sqlite3.Connection | None = None

    def _ensure_connected(self) -> sqlite3.Connection:
        """Ensure database connection and schema exist."""
        if self._conn is not None:
            return self._conn

        self._conn = sqlite3.connect(str(self.db_path))

        # Track fetched files
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS fetch_log (
                symbol TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                file_path TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                latest_annual_date TEXT,
                latest_quarterly_date TEXT,
                PRIMARY KEY (symbol, endpoint)
            )
        """)

        # Track API rate limit usage
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS api_calls (
                call_date TEXT NOT NULL,
                call_count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (call_date)
            )
        """)

        # Pending new filing: head newer than provider data
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS pending_new_filing (
                symbol TEXT NOT NULL PRIMARY KEY,
                first_pending_at TEXT NOT NULL
            )
        """)

        self._conn.commit()
        return self._conn

    def get_fetch_record(self, symbol: str, endpoint: str) -> FetchRecord | None:
        """Get the fetch record for a symbol/endpoint pair.

        Args:
            symbol: Stock ticker
            endpoint: "income_statement" or "balance_sheet"

        Returns:
            FetchRecord if exists, None otherwise
        """
        conn = self._ensure_connected()
        cursor = conn.execute(
            """
            SELECT symbol, endpoint, file_path, fetched_at,
                   latest_annual_date, latest_quarterly_date
            FROM fetch_log
            WHERE symbol = ? AND endpoint = ?
            """,
            (symbol, endpoint),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return FetchRecord(
            symbol=row[0],
            endpoint=row[1],
            file_path=row[2],
            fetched_at=row[3],
            latest_annual_date=row[4],
            latest_quarterly_date=row[5],
        )

    def record_fetch(
        self,
        symbol: str,
        endpoint: str,
        file_path: str,
        latest_annual_date: str | None,
        latest_quarterly_date: str | None,
    ) -> None:
        """Record that a file was fetched.

        Args:
            symbol: Stock ticker
            endpoint: "income_statement" or "balance_sheet"
            file_path: Path where JSON was saved
            latest_annual_date: Most recent annual report date
            latest_quarterly_date: Most recent quarterly report date
        """
        conn = self._ensure_connected()
        now = datetime.now(UTC).isoformat()
        conn.execute(
            """
            INSERT OR REPLACE INTO fetch_log
            (symbol, endpoint, file_path, fetched_at, latest_annual_date, latest_quarterly_date)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                symbol,
                endpoint,
                file_path,
                now,
                latest_annual_date,
                latest_quarterly_date,
            ),
        )
        conn.commit()

    def get_api_calls_today(self) -> int:
        """Get the number of API calls made today.

        Returns:
            Number of API calls made today
        """
        conn = self._ensure_connected()
        today = date.today().isoformat()
        cursor = conn.execute(
            "SELECT call_count FROM api_calls WHERE call_date = ?",
            (today,),
        )
        row = cursor.fetchone()
        return row[0] if row else 0

    def increment_api_calls(self, count: int = 1) -> int:
        """Increment the API call counter for today.

        Args:
            count: Number of calls to add

        Returns:
            New total for today
        """
        conn = self._ensure_connected()
        today = date.today().isoformat()

        # Use upsert
        conn.execute(
            """
            INSERT INTO api_calls (call_date, call_count)
            VALUES (?, ?)
            ON CONFLICT(call_date) DO UPDATE SET
            call_count = call_count + excluded.call_count
            """,
            (today, count),
        )
        conn.commit()

        return self.get_api_calls_today()

    def mark_pending_new_filing(
        self, symbol: str, *, first_pending_at: str | None = None
    ) -> str:
        """Persist first_pending_at for a symbol (idempotent keep-earliest)."""
        conn = self._ensure_connected()
        now = first_pending_at or datetime.now(UTC).isoformat()
        existing = self.get_pending_new_filing(symbol)
        if existing is not None:
            return existing
        conn.execute(
            """
            INSERT INTO pending_new_filing (symbol, first_pending_at)
            VALUES (?, ?)
            """,
            (symbol.upper(), now),
        )
        conn.commit()
        return now

    def clear_pending_new_filing(self, symbol: str) -> None:
        """Clear pending state after a successful head-matching pull."""
        conn = self._ensure_connected()
        conn.execute(
            "DELETE FROM pending_new_filing WHERE symbol = ?",
            (symbol.upper(),),
        )
        conn.commit()

    def get_pending_new_filing(self, symbol: str) -> str | None:
        """Return first_pending_at ISO string or None."""
        conn = self._ensure_connected()
        cursor = conn.execute(
            "SELECT first_pending_at FROM pending_new_filing WHERE symbol = ?",
            (symbol.upper(),),
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def get_all_fetched_symbols(self) -> list[str]:
        """Get all symbols that have been fetched.

        Returns:
            List of unique symbols
        """
        conn = self._ensure_connected()
        cursor = conn.execute("SELECT DISTINCT symbol FROM fetch_log")
        return [row[0] for row in cursor.fetchall()]

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

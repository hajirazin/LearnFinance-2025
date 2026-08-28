"""DuckDB repository for news events, coverage, jobs, and score cache."""

from __future__ import annotations

import fcntl
import logging
import threading
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from brain_api.news.errors import NewsCoverageMissing, NewsStoreConflict
from brain_api.news.models import (
    NEWS_PROVIDER,
    NEWS_SCHEMA_VERSION,
    NEWS_SENTIMENT_MODEL,
    NEWS_SENTIMENT_REVISION,
    NewsCoverage,
    NewsEvent,
    NewsJob,
    NewsWindow,
)

logger = logging.getLogger(__name__)

_WRITE_LOCK = threading.Lock()
_RETRY_BACKOFFS = (0.05, 0.1, 0.2)


def news_db_path(base_path: Path | None = None) -> Path:
    if base_path is None:
        from brain_api.storage.base import DEFAULT_DATA_PATH

        base_path = DEFAULT_DATA_PATH
    root = Path(base_path)
    return root / "news" / "news_v1.duckdb"


def news_lock_path(base_path: Path | None = None) -> Path:
    return news_db_path(base_path).with_suffix(".lock")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS news_events (
    provider VARCHAR NOT NULL,
    provider_article_id VARCHAR NOT NULL,
    symbol VARCHAR NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    source VARCHAR NOT NULL,
    sentiment_score DOUBLE NOT NULL,
    p_positive DOUBLE NOT NULL,
    p_negative DOUBLE NOT NULL,
    p_neutral DOUBLE NOT NULL,
    confidence DOUBLE NOT NULL,
    scored_text_sha256 VARCHAR NOT NULL,
    sentiment_model VARCHAR NOT NULL,
    sentiment_model_revision VARCHAR NOT NULL,
    schema_version INTEGER NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (provider, provider_article_id, symbol, updated_at)
);
CREATE TABLE IF NOT EXISTS news_coverage (
    provider VARCHAR NOT NULL,
    symbol VARCHAR NOT NULL,
    window_start_exclusive TIMESTAMPTZ NOT NULL,
    window_end_inclusive TIMESTAMPTZ NOT NULL,
    schema_version INTEGER NOT NULL,
    sentiment_model VARCHAR NOT NULL,
    sentiment_model_revision VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    page_count INTEGER NOT NULL,
    event_count INTEGER NOT NULL,
    future_revision_excluded_count INTEGER NOT NULL,
    fetched_at TIMESTAMPTZ NOT NULL,
    request_manifest_hash VARCHAR NOT NULL,
    PRIMARY KEY (
        provider, symbol, window_start_exclusive, window_end_inclusive,
        schema_version, sentiment_model, sentiment_model_revision
    )
);
CREATE TABLE IF NOT EXISTS news_jobs (
    job_id VARCHAR PRIMARY KEY,
    requested_start TIMESTAMPTZ NOT NULL,
    requested_end TIMESTAMPTZ NOT NULL,
    symbols_hash VARCHAR NOT NULL,
    schema_version INTEGER NOT NULL,
    sentiment_revision VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    last_completed_symbol VARCHAR,
    last_completed_window_end TIMESTAMPTZ,
    windows_done INTEGER NOT NULL,
    windows_total INTEGER NOT NULL,
    events_scored INTEGER NOT NULL,
    error VARCHAR,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);
CREATE TABLE IF NOT EXISTS news_score_cache (
    scored_text_sha256 VARCHAR NOT NULL,
    sentiment_model VARCHAR NOT NULL,
    sentiment_model_revision VARCHAR NOT NULL,
    scoring_schema_version INTEGER NOT NULL,
    sentiment_score DOUBLE NOT NULL,
    p_positive DOUBLE NOT NULL,
    p_negative DOUBLE NOT NULL,
    p_neutral DOUBLE NOT NULL,
    confidence DOUBLE NOT NULL,
    PRIMARY KEY (
        scored_text_sha256, sentiment_model,
        sentiment_model_revision, scoring_schema_version
    )
);
"""


class NewsStore:
    """Device-local DuckDB. Writes are flock + threading.Lock serialized."""

    def __init__(self, base_path: Path | None = None) -> None:
        self.base_path = news_db_path(base_path).parent.parent
        self.db_path = news_db_path(base_path)
        self.lock_path = news_lock_path(base_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.db_path))

    def _ensure_schema(self) -> None:
        with self._connect() as con:
            con.execute(_SCHEMA)

    def _with_write_lock(self, fn):
        last_error: Exception | None = None
        for attempt, backoff in enumerate((*_RETRY_BACKOFFS, None)):
            with _WRITE_LOCK:
                self.lock_path.parent.mkdir(parents=True, exist_ok=True)
                self.lock_path.touch(exist_ok=True)
                with self.lock_path.open("a") as handle:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                    try:
                        with self._connect() as con:
                            con.execute("BEGIN")
                            try:
                                result = fn(con)
                                con.execute("COMMIT")
                                return result
                            except Exception:
                                con.execute("ROLLBACK")
                                raise
                    except duckdb.TransactionException as exc:
                        last_error = exc
                    finally:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            if backoff is None:
                break
            logger.warning(
                "DuckDB conflict attempt=%s backoff=%.2fs", attempt + 1, backoff
            )
            time.sleep(backoff)
        raise NewsStoreConflict("DuckDB writer conflict after retries") from last_error

    def cache_get(
        self,
        scored_text_sha256: str,
        *,
        sentiment_model: str = NEWS_SENTIMENT_MODEL,
        sentiment_model_revision: str = NEWS_SENTIMENT_REVISION,
        scoring_schema_version: int = 1,
    ) -> tuple[float, float, float, float, float] | None:
        with self._connect() as con:
            row = con.execute(
                """
                SELECT sentiment_score, p_positive, p_negative, p_neutral, confidence
                FROM news_score_cache
                WHERE scored_text_sha256 = ?
                  AND sentiment_model = ?
                  AND sentiment_model_revision = ?
                  AND scoring_schema_version = ?
                """,
                [
                    scored_text_sha256,
                    sentiment_model,
                    sentiment_model_revision,
                    scoring_schema_version,
                ],
            ).fetchone()
        if row is None:
            return None
        return tuple(float(value) for value in row)  # type: ignore[return-value]

    def commit_window(
        self,
        *,
        events: Sequence[NewsEvent],
        coverage: NewsCoverage,
        cache_rows: Sequence[
            tuple[str, str, str, int, float, float, float, float, float]
        ],
    ) -> None:
        def _write(con: duckdb.DuckDBPyConnection) -> None:
            for event in events:
                con.execute(
                    """
                    INSERT OR REPLACE INTO news_events VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    [
                        event.provider,
                        event.provider_article_id,
                        event.symbol,
                        event.created_at,
                        event.updated_at,
                        event.source,
                        event.sentiment_score,
                        event.p_positive,
                        event.p_negative,
                        event.p_neutral,
                        event.confidence,
                        event.scored_text_sha256,
                        event.sentiment_model,
                        event.sentiment_model_revision,
                        event.schema_version,
                        event.ingested_at,
                    ],
                )
            con.execute(
                """
                INSERT OR REPLACE INTO news_coverage VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                [
                    coverage.provider,
                    coverage.symbol,
                    coverage.window_start_exclusive,
                    coverage.window_end_inclusive,
                    coverage.schema_version,
                    coverage.sentiment_model,
                    coverage.sentiment_model_revision,
                    coverage.status,
                    coverage.page_count,
                    coverage.event_count,
                    coverage.future_revision_excluded_count,
                    coverage.fetched_at,
                    coverage.request_manifest_hash,
                ],
            )
            for row in cache_rows:
                con.execute(
                    """
                    INSERT OR REPLACE INTO news_score_cache VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    list(row),
                )

        self._with_write_lock(_write)

    def get_coverage(
        self,
        symbol: str,
        window: NewsWindow,
        *,
        provider: str = NEWS_PROVIDER,
        schema_version: int = NEWS_SCHEMA_VERSION,
        sentiment_model: str = NEWS_SENTIMENT_MODEL,
        sentiment_model_revision: str = NEWS_SENTIMENT_REVISION,
    ) -> NewsCoverage | None:
        with self._connect() as con:
            row = con.execute(
                """
                SELECT provider, symbol, window_start_exclusive, window_end_inclusive,
                       schema_version, sentiment_model, sentiment_model_revision,
                       status, page_count, event_count, future_revision_excluded_count,
                       fetched_at, request_manifest_hash
                FROM news_coverage
                WHERE provider = ? AND symbol = ?
                  AND window_start_exclusive = ? AND window_end_inclusive = ?
                  AND schema_version = ? AND sentiment_model = ?
                  AND sentiment_model_revision = ?
                """,
                [
                    provider,
                    symbol,
                    window.start_exclusive,
                    window.end_inclusive,
                    schema_version,
                    sentiment_model,
                    sentiment_model_revision,
                ],
            ).fetchone()
        if row is None:
            return None
        return NewsCoverage(
            provider=row[0],
            symbol=row[1],
            window_start_exclusive=row[2],
            window_end_inclusive=row[3],
            schema_version=int(row[4]),
            sentiment_model=row[5],
            sentiment_model_revision=row[6],
            status=row[7],
            page_count=int(row[8]),
            event_count=int(row[9]),
            future_revision_excluded_count=int(row[10]),
            fetched_at=row[11],
            request_manifest_hash=row[12],
        )

    def coverage_keys(
        self,
        symbols: Sequence[str],
        *,
        provider: str = NEWS_PROVIDER,
        schema_version: int = NEWS_SCHEMA_VERSION,
        sentiment_model: str = NEWS_SENTIMENT_MODEL,
        sentiment_model_revision: str = NEWS_SENTIMENT_REVISION,
    ) -> set[tuple[str, datetime, datetime]]:
        """All coverage PK instants for these symbols under the current news identity."""
        if not symbols:
            return set()
        placeholders = ", ".join("?" * len(symbols))
        sql = f"""
            SELECT symbol, window_start_exclusive, window_end_inclusive
            FROM news_coverage
            WHERE provider = ?
              AND schema_version = ?
              AND sentiment_model = ?
              AND sentiment_model_revision = ?
              AND symbol IN ({placeholders})
        """
        params: list[object] = [
            provider,
            schema_version,
            sentiment_model,
            sentiment_model_revision,
            *symbols,
        ]
        with self._connect() as con:
            rows = con.execute(sql, params).fetchall()
        return {coverage_key(row[0], row[1], row[2]) for row in rows}

    def covered_symbols(
        self,
        symbols: Sequence[str],
        window: NewsWindow,
        *,
        provider: str = NEWS_PROVIDER,
        schema_version: int = NEWS_SCHEMA_VERSION,
        sentiment_model: str = NEWS_SENTIMENT_MODEL,
        sentiment_model_revision: str = NEWS_SENTIMENT_REVISION,
    ) -> set[str]:
        """Symbols in ``symbols`` that already have coverage for this exact window."""
        if not symbols:
            return set()
        placeholders = ", ".join("?" * len(symbols))
        sql = f"""
            SELECT symbol
            FROM news_coverage
            WHERE provider = ?
              AND symbol IN ({placeholders})
              AND window_start_exclusive = ?
              AND window_end_inclusive = ?
              AND schema_version = ?
              AND sentiment_model = ?
              AND sentiment_model_revision = ?
        """
        params: list[object] = [
            provider,
            *symbols,
            window.start_exclusive,
            window.end_inclusive,
            schema_version,
            sentiment_model,
            sentiment_model_revision,
        ]
        with self._connect() as con:
            rows = con.execute(sql, params).fetchall()
        return {row[0] for row in rows}

    def query_events(
        self,
        symbols: Sequence[str],
        window: NewsWindow,
        *,
        provider: str = NEWS_PROVIDER,
    ) -> list[NewsEvent]:
        if not symbols:
            return []
        placeholders = ", ".join("?" * len(symbols))
        sql = f"""
            WITH ranked AS (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY provider, provider_article_id, symbol
                        ORDER BY updated_at DESC
                    ) AS rn
                FROM news_events
                WHERE provider = ?
                  AND symbol IN ({placeholders})
                  AND updated_at <= ?
            )
            SELECT provider, provider_article_id, symbol, created_at, updated_at,
                   source, sentiment_score, p_positive, p_negative, p_neutral,
                   confidence, scored_text_sha256, sentiment_model,
                   sentiment_model_revision, schema_version, ingested_at
            FROM ranked
            WHERE rn = 1
              AND created_at > ?
              AND created_at <= ?
            ORDER BY symbol, created_at, provider_article_id
        """
        params: list[object] = [provider, *symbols, window.end_inclusive]
        params.extend([window.start_exclusive, window.end_inclusive])
        with self._connect() as con:
            rows = con.execute(sql, params).fetchall()
        events: list[NewsEvent] = []
        for row in rows:
            events.append(
                NewsEvent(
                    provider=row[0],
                    provider_article_id=row[1],
                    symbol=row[2],
                    created_at=row[3],
                    updated_at=row[4],
                    source=row[5],
                    sentiment_score=float(row[6]),
                    p_positive=float(row[7]),
                    p_negative=float(row[8]),
                    p_neutral=float(row[9]),
                    confidence=float(row[10]),
                    scored_text_sha256=row[11],
                    sentiment_model=row[12],
                    sentiment_model_revision=row[13],
                    schema_version=int(row[14]),
                    ingested_at=row[15],
                )
            )
        return events

    def require_coverage(
        self, symbols: Sequence[str], window: NewsWindow
    ) -> list[NewsCoverage]:
        rows: list[NewsCoverage] = []
        missing: list[str] = []
        for symbol in symbols:
            coverage = self.get_coverage(symbol, window)
            if coverage is None:
                missing.append(symbol)
            else:
                rows.append(coverage)
        if missing:
            raise NewsCoverageMissing(
                f"missing news coverage for {missing} window "
                f"{window.start_exclusive.isoformat()}->{window.end_inclusive.isoformat()}"
            )
        return rows

    def upsert_job(self, job: NewsJob) -> None:
        def _write(con: duckdb.DuckDBPyConnection) -> None:
            con.execute(
                """
                INSERT OR REPLACE INTO news_jobs VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                [
                    job.job_id,
                    job.requested_start,
                    job.requested_end,
                    job.symbols_hash,
                    job.schema_version,
                    job.sentiment_revision,
                    job.status,
                    job.last_completed_symbol,
                    job.last_completed_window_end,
                    job.windows_done,
                    job.windows_total,
                    job.events_scored,
                    job.error,
                    job.created_at,
                    job.updated_at,
                ],
            )

        self._with_write_lock(_write)

    def get_job(self, job_id: str) -> NewsJob | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT * FROM news_jobs WHERE job_id = ?", [job_id]
            ).fetchone()
        if row is None:
            return None
        return NewsJob(
            job_id=row[0],
            requested_start=row[1],
            requested_end=row[2],
            symbols_hash=row[3],
            schema_version=int(row[4]),
            sentiment_revision=row[5],
            status=row[6],
            last_completed_symbol=row[7],
            last_completed_window_end=row[8],
            windows_done=int(row[9]),
            windows_total=int(row[10]),
            events_scored=int(row[11]),
            error=row[12],
            created_at=row[13],
            updated_at=row[14],
        )


def coverage_key(
    symbol: str, start: datetime, end: datetime
) -> tuple[str, datetime, datetime]:
    """UTC-normalized coverage membership key for one ``(symbol, window)``."""
    return (symbol, start.astimezone(UTC), end.astimezone(UTC))


def utcnow() -> datetime:
    return datetime.now(tz=UTC)

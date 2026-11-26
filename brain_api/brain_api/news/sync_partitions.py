"""Typed partitions for transferring local news evidence as Parquet."""

import re
from pathlib import Path

from brain_api.news.models import (
    NEWS_PROVIDER,
    NEWS_SCHEMA_VERSION,
    NEWS_SENTIMENT_MODEL,
    NEWS_SENTIMENT_REVISION,
    SCORING_SCHEMA_VERSION,
)

_IDENTITY = (
    f"sentiment_model = '{NEWS_SENTIMENT_MODEL}' AND "
    f"sentiment_model_revision = '{NEWS_SENTIMENT_REVISION}'"
)
_EVENTS = f"{_IDENTITY} AND provider = '{NEWS_PROVIDER}' AND schema_version = {NEWS_SCHEMA_VERSION}"
_TABLES = {
    "news_events": ("strftime(created_at AT TIME ZONE 'UTC', '%Y-%m')", _EVENTS),
    "news_coverage": (
        "strftime(window_end_inclusive AT TIME ZONE 'UTC', '%Y-%m')",
        f"{_EVENTS} AND status IN ('complete', 'verified_empty')",
    ),
    "news_score_cache": (
        "left(scored_text_sha256, 2)",
        f"{_IDENTITY} AND scoring_schema_version = {SCORING_SCHEMA_VERSION}",
    ),
}
_PARTITION = re.compile(
    r"(?:news_events|news_coverage)/[0-9]{4}-(?:0[1-9]|1[0-2])|"
    r"news_score_cache/[0-9a-f]{2}"
)


def mark_dirty(con, partitions):
    if partitions:
        con.executemany(
            "INSERT INTO news_sync_dirty VALUES (?, 1) ON CONFLICT (partition) DO UPDATE "
            "SET generation = news_sync_dirty.generation + 1",
            [(partition,) for partition in sorted(partitions)],
        )


def seed_local_partitions(con, repo: str) -> None:
    """Index evidence from databases created before sync was installed."""
    if con.execute("SELECT 1 FROM news_sync_seeded WHERE repo = ?", [repo]).fetchone():
        return
    for table, (expression, identity) in _TABLES.items():
        rows = con.execute(
            f"SELECT DISTINCT {expression} FROM {table} WHERE {identity}"
        ).fetchall()
        mark_dirty(con, {f"{table}/{key}" for (key,) in rows})
    con.execute("INSERT INTO news_sync_seeded VALUES (?)", [repo])


def partition_query(partition: str) -> tuple[str, str]:
    if not _PARTITION.fullmatch(partition):
        raise ValueError(f"Invalid news partition {partition!r}")
    table, key = partition.split("/")
    expression, identity = _TABLES[table]
    return table, f"{identity} AND {expression} = '{key}'"


def export_partition(con, partition: str, path: Path) -> int:
    table, where = partition_query(partition)
    # COPY does not accept a bound filename in DuckDB's prepared statements.
    escaped = str(path).replace("'", "''")
    return con.execute(
        f"COPY (SELECT * FROM {table} WHERE {where} ORDER BY ALL) "
        f"TO '{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    ).fetchone()[0]


def import_partition(con, partition: str, path: Path, expected_rows: int) -> None:
    table, where = partition_query(partition)
    columns = [
        row[1] for row in con.execute(f"PRAGMA table_info('{table}')").fetchall()
    ]
    actual_columns = [
        row[0]
        for row in con.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]
        ).fetchall()
    ]
    if columns != actual_columns:
        raise ValueError(f"Unexpected columns for news partition {partition}")
    counts = con.execute(
        f"SELECT count(*), count(*) FILTER (WHERE {where}) FROM read_parquet(?)",
        [str(path)],
    ).fetchone()
    if counts != (expected_rows, expected_rows):
        raise ValueError(
            f"Unexpected row count or identity for news partition {partition}"
        )
    keys = {
        "news_events": ["provider", "provider_article_id", "symbol", "updated_at"],
        "news_coverage": [
            "provider",
            "symbol",
            "window_start_exclusive",
            "window_end_inclusive",
            "schema_version",
            "sentiment_model",
            "sentiment_model_revision",
        ],
        "news_score_cache": [
            "scored_text_sha256",
            "sentiment_model",
            "sentiment_model_revision",
            "scoring_schema_version",
        ],
    }[table]
    # Prefer the coverage with the most usable historical evidence. Remaining
    # fields break ties deterministically, independent of which host imports.
    quality = (
        ["future_revision_excluded_count ASC", "event_count DESC", "fetched_at DESC"]
        if table == "news_coverage"
        else []
    )
    order = quality + [f'"{column}" DESC' for column in columns if column not in keys]
    con.execute(
        f"""
        INSERT OR REPLACE INTO {table}
        SELECT * FROM (
            SELECT * FROM {table} WHERE {where}
            UNION ALL SELECT * FROM read_parquet(?)
        ) merged
        QUALIFY row_number() OVER (
            PARTITION BY {", ".join(keys)} ORDER BY {", ".join(order)}
        ) = 1
        """,
        [str(path)],
    )
    local_only = con.execute(
        f"SELECT count(*) FROM (SELECT * FROM {table} WHERE {where} "
        "EXCEPT SELECT * FROM read_parquet(?))",
        [str(path)],
    ).fetchone()[0]
    if local_only:
        mark_dirty(con, {partition})

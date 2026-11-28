"""Find existing complete Monday coverage using a read-only DB connection."""

import json
from pathlib import Path

import duckdb

from brain_api.news.models import (
    NEWS_PROVIDER,
    NEWS_SCHEMA_VERSION,
    NEWS_SENTIMENT_MODEL,
    NEWS_SENTIMENT_REVISION,
)

root = Path("/app/data")
model_root = root / "models" / "sac_halal_filtered"
version = (model_root / "current").read_text().strip()
metadata = json.loads((model_root / version / "metadata.json").read_text())
symbols = metadata.get("symbol_order", metadata.get("symbols"))
if not symbols:
    raise RuntimeError("Active artifact lacks symbol order")

with duckdb.connect(str(root / "news/news_v1.duckdb"), read_only=True) as con:
    query = """
    SELECT window_start_exclusive, window_end_inclusive,
           count(DISTINCT symbol) AS covered_symbols,
           list(DISTINCT symbol ORDER BY symbol) AS symbols
    FROM news_coverage
    WHERE symbol IN (SELECT unnest(?)) AND provider = ? AND schema_version = ?
      AND sentiment_model = ? AND sentiment_model_revision = ?
      AND status IN ('complete', 'verified_empty')
      AND window_end_inclusive > TIMESTAMPTZ '2024-11-15 00:00:00+00'
      AND isodow(timezone('America/New_York', window_end_inclusive)) = 1
      AND hour(timezone('America/New_York', window_end_inclusive)) = 9
      AND minute(timezone('America/New_York', window_end_inclusive)) = 0
    GROUP BY 1,2 ORDER BY covered_symbols DESC, window_end_inclusive DESC LIMIT 12
    """
    cur = con.execute(
        query,
        [
            symbols,
            NEWS_PROVIDER,
            NEWS_SCHEMA_VERSION,
            NEWS_SENTIMENT_MODEL,
            NEWS_SENTIMENT_REVISION,
        ],
    )
    cols = [d[0] for d in cur.description]
    print(
        json.dumps(
            {
                "check": "candidate_mondays",
                "symbols_required": len(symbols),
                "windows": [dict(zip(cols, row)) for row in cur.fetchall()],
            },
            default=str,
        ),
        flush=True,
    )
    for label, query in (
        (
            "events",
            "SELECT count(*) AS rows, count(DISTINCT symbol) AS symbols, min(created_at) AS first_event, max(created_at) AS last_event FROM news_events",
        ),
        (
            "coverage",
            "SELECT count(*) AS rows, count(DISTINCT symbol) AS symbols, min(window_start_exclusive) AS first_start, max(window_end_inclusive) AS last_end FROM news_coverage",
        ),
        ("imported_partitions", "SELECT count(*) AS partitions FROM news_sync_imports"),
    ):
        cur = con.execute(query)
        cols = [d[0] for d in cur.description]
        print(
            json.dumps(
                {
                    "check": label,
                    "result": [dict(zip(cols, row)) for row in cur.fetchall()],
                },
                default=str,
            ),
            flush=True,
        )

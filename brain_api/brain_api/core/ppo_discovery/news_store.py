"""PPO weekly news parquet: first-write-wins per cutoff, replace on force."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from brain_api.core.ppo_discovery.config import NEWS_RECENCY_TAU_HOURS
from brain_api.core.ppo_discovery.news_evidence import (
    FINBERT_REVISION,
    PROVIDER_NAME,
    NewsEvidenceError,
    NewsWindow,
)
from brain_api.core.ppo_discovery.schemas import SymbolNewsFeatures
from brain_api.storage.base import DEFAULT_DATA_PATH

PARQUET_RELATIVE = Path("ppo_discovery") / "news" / "weekly_features.parquet"


def weekly_news_path(base_path: Path | str | None = None) -> Path:
    return Path(base_path or DEFAULT_DATA_PATH) / PARQUET_RELATIVE


def persist_weekly_news_features(
    cutoff: datetime,
    features: dict[str, SymbolNewsFeatures],
    *,
    window: NewsWindow,
    base_path: Path | str | None = None,
    force: bool = False,
) -> Path:
    """Write one decision-cutoff partition. Existing cutoff is skipped unless ``force``."""
    path = weekly_news_path(base_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cutoff_iso = cutoff.astimezone(UTC).isoformat()
    rows = [
        {
            "decision_cutoff": cutoff_iso,
            "symbol": symbol,
            "window_start": window.start.isoformat(),
            "window_end": window.end.isoformat(),
            "raw_sentiment": row.raw_sentiment,
            "article_count": row.article_count,
            "average_confidence": row.average_confidence,
            "sentiment_dispersion": row.sentiment_dispersion,
            "hours_since_latest": row.hours_since_latest,
            "unique_source_count": row.unique_source_count,
            "has_news": row.has_news,
            "query_complete": int(row.query_complete),
            "provider": PROVIDER_NAME,
            "finbert_revision": FINBERT_REVISION,
            "article_ids_sha256": row.article_ids_sha256,
            "request_manifest_sha256": row.request_manifest_sha256,
        }
        for symbol, row in sorted(features.items())
    ]
    incoming = pa.Table.from_pylist(rows)
    if path.exists():
        existing = pq.read_table(path).to_pandas()
        same = existing["decision_cutoff"] == cutoff_iso
        if bool(same.any()) and not force:
            return path
        kept = existing.loc[~same]
        if kept.empty:
            table = incoming
        else:
            table = pa.concat_tables(
                [pa.Table.from_pandas(kept, preserve_index=False), incoming]
            )
    else:
        table = incoming
    pq.write_table(table, path)
    return path


def load_weekly_news_features(
    cutoff: datetime,
    symbols: list[str],
    *,
    base_path: Path | str | None = None,
) -> dict[str, SymbolNewsFeatures]:
    """Load one complete partition. Missing symbol or incomplete query aborts."""
    path = weekly_news_path(base_path)
    if not path.exists():
        raise NewsEvidenceError("ppo_discovery weekly news parquet is missing")
    cutoff_iso = cutoff.astimezone(UTC).isoformat()
    frame = pq.read_table(path).to_pandas()
    part = frame.loc[frame["decision_cutoff"] == cutoff_iso]
    if part.empty:
        raise NewsEvidenceError(f"no news partition for {cutoff_iso}")
    by_symbol = {str(row.symbol): row for row in part.itertuples()}
    features: dict[str, SymbolNewsFeatures] = {}
    for symbol in symbols:
        row = by_symbol.get(symbol)
        if row is None:
            raise NewsEvidenceError(f"news partition missing {symbol} at {cutoff_iso}")
        if int(row.query_complete) != 1:
            raise NewsEvidenceError(
                f"incomplete news query for {symbol} at {cutoff_iso}"
            )
        features[symbol] = SymbolNewsFeatures(
            symbol=symbol,
            raw_sentiment=float(row.raw_sentiment),
            article_count=int(row.article_count),
            average_confidence=float(row.average_confidence),
            sentiment_dispersion=float(row.sentiment_dispersion),
            hours_since_latest=float(row.hours_since_latest),
            unique_source_count=int(row.unique_source_count),
            has_news=int(row.has_news),
            query_complete=True,
            news_recency=0.0
            if int(row.article_count) == 0
            else float(
                math.exp(-float(row.hours_since_latest) / NEWS_RECENCY_TAU_HOURS)
            ),
            log1p_article_count=float(math.log1p(int(row.article_count))),
            article_ids_sha256=str(row.article_ids_sha256),
            request_manifest_sha256=str(row.request_manifest_sha256),
        )
    return features


def partition_exists(cutoff: datetime, *, base_path: Path | str | None = None) -> bool:
    path = weekly_news_path(base_path)
    if not path.exists():
        return False
    cutoff_iso = cutoff.astimezone(UTC).isoformat()
    frame = pq.read_table(path).to_pandas()
    return bool((frame["decision_cutoff"] == cutoff_iso).any())


__all__ = [
    "load_weekly_news_features",
    "partition_exists",
    "persist_weekly_news_features",
    "weekly_news_path",
]

"""Canonical hashes for news coverage and ETL job identity."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import datetime

from brain_api.news.models import (
    NEWS_PROVIDER,
    NEWS_SCHEMA_VERSION,
    NEWS_SENTIMENT_MODEL,
    NEWS_SENTIMENT_REVISION,
    NewsWindow,
)


def _canonical_instant(value: datetime) -> str:
    return value.isoformat()


def canonical_json_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def request_manifest_hash(
    symbols: Sequence[str],
    window: NewsWindow,
    *,
    schema_version: int = NEWS_SCHEMA_VERSION,
    sentiment_model: str = NEWS_SENTIMENT_MODEL,
    sentiment_model_revision: str = NEWS_SENTIMENT_REVISION,
    provider: str = NEWS_PROVIDER,
) -> str:
    return canonical_json_hash(
        {
            "provider": provider,
            "symbols": sorted(symbols),
            "start_exclusive": _canonical_instant(window.start_exclusive),
            "end_inclusive": _canonical_instant(window.end_inclusive),
            "schema_version": schema_version,
            "sentiment_model": sentiment_model,
            "sentiment_model_revision": sentiment_model_revision,
        }
    )


def news_job_id(
    *,
    start: datetime,
    end: datetime,
    symbols: Sequence[str],
    schema_version: int = NEWS_SCHEMA_VERSION,
    sentiment_model: str = NEWS_SENTIMENT_MODEL,
    sentiment_model_revision: str = NEWS_SENTIMENT_REVISION,
) -> str:
    return canonical_json_hash(
        {
            "start": _canonical_instant(start),
            "end": _canonical_instant(end),
            "symbols": sorted(symbols),
            "schema_version": schema_version,
            "sentiment_model": sentiment_model,
            "sentiment_model_revision": sentiment_model_revision,
        }
    )


def symbols_hash(symbols: Sequence[str]) -> str:
    return canonical_json_hash({"symbols": sorted(symbols)})

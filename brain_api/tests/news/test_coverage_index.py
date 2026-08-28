from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from brain_api.news.models import (
    NEWS_PROVIDER,
    NEWS_SCHEMA_VERSION,
    NEWS_SENTIMENT_MODEL,
    NEWS_SENTIMENT_REVISION,
    NewsCoverage,
    NewsWindow,
)
from brain_api.news.store import NewsStore, coverage_key, utcnow

NY = ZoneInfo("America/New_York")


def _window() -> NewsWindow:
    return NewsWindow(
        start_exclusive=datetime(2026, 8, 17, 9, 0, tzinfo=NY),
        end_inclusive=datetime(2026, 8, 24, 9, 0, tzinfo=NY),
    )


def _coverage(
    symbol: str, window: NewsWindow, *, revision: str | None = None
) -> NewsCoverage:
    return NewsCoverage(
        provider=NEWS_PROVIDER,
        symbol=symbol,
        window_start_exclusive=window.start_exclusive,
        window_end_inclusive=window.end_inclusive,
        schema_version=NEWS_SCHEMA_VERSION,
        sentiment_model=NEWS_SENTIMENT_MODEL,
        sentiment_model_revision=revision or NEWS_SENTIMENT_REVISION,
        status="complete",
        page_count=1,
        event_count=0,
        future_revision_excluded_count=0,
        fetched_at=utcnow(),
        request_manifest_hash="m",
    )


def test_coverage_keys_empty_symbols(tmp_path) -> None:
    store = NewsStore(tmp_path)
    assert store.coverage_keys([]) == set()
    assert store.covered_symbols([], _window()) == set()


def test_coverage_keys_contains_committed_window_in_utc(tmp_path) -> None:
    store = NewsStore(tmp_path)
    window = _window()
    store.commit_window(events=[], coverage=_coverage("AAPL", window), cache_rows=[])
    keys = store.coverage_keys(["AAPL", "MSFT"])
    expected = coverage_key("AAPL", window.start_exclusive, window.end_inclusive)
    assert expected in keys
    assert not any(symbol == "MSFT" for symbol, _start, _end in keys)


def test_coverage_keys_ignores_other_sentiment_revision(tmp_path) -> None:
    store = NewsStore(tmp_path)
    window = _window()
    store.commit_window(
        events=[],
        coverage=_coverage("AAPL", window, revision="deadbeef"),
        cache_rows=[],
    )
    assert store.coverage_keys(["AAPL"]) == set()
    assert store.covered_symbols(["AAPL"], window) == set()


def test_cache_get_many_returns_committed_digests(tmp_path) -> None:
    store = NewsStore(tmp_path)
    window = _window()
    digest = "a" * 64
    store.commit_window(
        events=[],
        coverage=_coverage("AAPL", window),
        cache_rows=[
            (
                digest,
                NEWS_SENTIMENT_MODEL,
                NEWS_SENTIMENT_REVISION,
                1,
                0.2,
                0.6,
                0.3,
                0.1,
                0.5,
            )
        ],
    )
    assert store.cache_get_many([]) == {}
    hits = store.cache_get_many([digest, "b" * 64])
    assert hits[digest] == (0.2, 0.6, 0.3, 0.1, 0.5)
    assert "b" * 64 not in hits
    assert store.cache_get(digest) == hits[digest]


def test_covered_symbols_returns_only_this_window(tmp_path) -> None:
    store = NewsStore(tmp_path)
    window = _window()
    store.commit_window(events=[], coverage=_coverage("AAPL", window), cache_rows=[])
    assert store.covered_symbols(["AAPL", "MSFT"], window) == {"AAPL"}


def test_news_store_rejects_live_default_path() -> None:
    with pytest.raises(RuntimeError, match="live news DuckDB"):
        NewsStore()

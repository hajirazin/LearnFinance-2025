"""SAC weekly news loaders must use one coverage-grid read, not per-cell lookups."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from brain_api.core.portfolio_rl.data_loading import (
    load_weekly_news_scores,
    missing_weekly_news_coverage,
)
from brain_api.core.sac.news_adapter import build_sac_news_features
from brain_api.core.weekly_decision import (
    monday_cutoff_for_actor_friday,
    monday_window_bounds,
)
from brain_api.news.errors import NewsCoverageMissing
from brain_api.news.models import (
    NEWS_PROVIDER,
    NEWS_SCHEMA_VERSION,
    NEWS_SENTIMENT_MODEL,
    NEWS_SENTIMENT_REVISION,
    NewsCoverage,
    NewsEvent,
    NewsWindow,
)
from brain_api.news.store import NewsStore, utcnow

NY = ZoneInfo("America/New_York")
FRIDAYS = pd.DatetimeIndex(["2026-08-21", "2026-08-28"])


def _window_for_friday(friday: datetime) -> tuple[datetime, NewsWindow]:
    cutoff = monday_cutoff_for_actor_friday(friday.date())
    start_exclusive, end_inclusive = monday_window_bounds(cutoff.date())
    return cutoff, NewsWindow(
        start_exclusive=start_exclusive, end_inclusive=end_inclusive
    )


def _coverage(symbol: str, window: NewsWindow) -> NewsCoverage:
    return NewsCoverage(
        provider=NEWS_PROVIDER,
        symbol=symbol,
        window_start_exclusive=window.start_exclusive,
        window_end_inclusive=window.end_inclusive,
        schema_version=NEWS_SCHEMA_VERSION,
        sentiment_model=NEWS_SENTIMENT_MODEL,
        sentiment_model_revision=NEWS_SENTIMENT_REVISION,
        status="complete",
        page_count=1,
        event_count=1,
        future_revision_excluded_count=0,
        fetched_at=utcnow(),
        request_manifest_hash="m",
    )


def _event(
    symbol: str,
    window: NewsWindow,
    *,
    score: float,
    article_id: str,
) -> NewsEvent:
    created = window.start_exclusive + timedelta(days=1)
    return NewsEvent(
        provider=NEWS_PROVIDER,
        provider_article_id=article_id,
        symbol=symbol,
        created_at=created,
        updated_at=created,
        source="benzinga",
        sentiment_score=score,
        p_positive=0.6,
        p_negative=0.1,
        p_neutral=0.3,
        confidence=0.8,
        scored_text_sha256="abc",
        sentiment_model=NEWS_SENTIMENT_MODEL,
        sentiment_model_revision=NEWS_SENTIMENT_REVISION,
        schema_version=NEWS_SCHEMA_VERSION,
        ingested_at=utcnow(),
    )


def _seed_two_weeks(store: NewsStore) -> None:
    scores = {
        ("AAPL", 0): 0.2,
        ("MSFT", 0): -0.1,
        ("AAPL", 1): 0.7,
        ("MSFT", 1): 0.4,
    }
    for week_idx, friday in enumerate(FRIDAYS):
        _cutoff, window = _window_for_friday(friday)
        for symbol in ("AAPL", "MSFT"):
            store.commit_window(
                events=[
                    _event(
                        symbol,
                        window,
                        score=scores[(symbol, week_idx)],
                        article_id=f"{symbol}-{week_idx}",
                    )
                ],
                coverage=_coverage(symbol, window),
                cache_rows=[],
            )


def _per_week_news_scores(
    store: NewsStore, symbols: list[str], weekly_cutoffs: pd.DatetimeIndex
) -> dict[str, np.ndarray]:
    """Oracle: one require_coverage + query_events per week (legacy path)."""
    scores: dict[str, list[float]] = {symbol: [] for symbol in symbols}
    for timestamp in weekly_cutoffs:
        cutoff, window = _window_for_friday(timestamp)
        coverage = store.require_coverage(symbols, window)
        events = store.query_events(symbols, window)
        events_by_symbol: dict[str, list[NewsEvent]] = {
            symbol: [] for symbol in symbols
        }
        for event in events:
            if event.symbol in events_by_symbol:
                events_by_symbol[event.symbol].append(event)
        status = {row.symbol: row.status for row in coverage}
        week_scores = build_sac_news_features(
            events_by_symbol, cutoff=cutoff, coverage_status=status
        )
        for symbol in symbols:
            scores[symbol].append(week_scores[symbol])
    return {
        symbol: np.asarray(values, dtype=float) for symbol, values in scores.items()
    }


def test_missing_weekly_news_coverage_returns_empty_when_complete(tmp_path) -> None:
    store = NewsStore(tmp_path)
    _seed_two_weeks(store)
    assert missing_weekly_news_coverage(["AAPL", "MSFT"], FRIDAYS, store=store) == []


def test_missing_weekly_news_coverage_returns_every_missing_symbol(tmp_path) -> None:
    store = NewsStore(tmp_path)
    _cutoff, first = _window_for_friday(FRIDAYS[0])
    store.commit_window(events=[], coverage=_coverage("AAPL", first), cache_rows=[])

    missing = missing_weekly_news_coverage(["AAPL", "MSFT"], FRIDAYS, store=store)
    assert {symbol for symbol, _start, _end in missing} == {"AAPL", "MSFT"}
    assert len(missing) == 3


def test_load_weekly_news_scores_raises_when_a_cell_is_missing(tmp_path) -> None:
    store = NewsStore(tmp_path)
    _cutoff, first = _window_for_friday(FRIDAYS[0])
    store.commit_window(events=[], coverage=_coverage("AAPL", first), cache_rows=[])

    with pytest.raises(NewsCoverageMissing):
        load_weekly_news_scores(["AAPL", "MSFT"], FRIDAYS, store=store)


def test_load_weekly_news_scores_match_per_week_reads(tmp_path, monkeypatch) -> None:
    store = NewsStore(tmp_path)
    _seed_two_weeks(store)
    symbols = ["AAPL", "MSFT"]
    expected = _per_week_news_scores(store, symbols, FRIDAYS)

    calls = {"get_coverage": 0, "query_events": 0, "events_many": 0, "grid": 0}
    original_get = store.get_coverage
    original_query = store.query_events
    original_many = store.query_events_many
    original_grid = store.require_coverage_many

    def _get(*args, **kwargs):
        calls["get_coverage"] += 1
        return original_get(*args, **kwargs)

    def _query(*args, **kwargs):
        calls["query_events"] += 1
        return original_query(*args, **kwargs)

    def _events_many(*args, **kwargs):
        calls["events_many"] += 1
        return original_many(*args, **kwargs)

    def _grid(*args, **kwargs):
        calls["grid"] += 1
        return original_grid(*args, **kwargs)

    monkeypatch.setattr(store, "get_coverage", _get)
    monkeypatch.setattr(store, "query_events", _query)
    monkeypatch.setattr(store, "query_events_many", _events_many)
    monkeypatch.setattr(store, "require_coverage_many", _grid)

    actual = load_weekly_news_scores(symbols, FRIDAYS, store=store)

    assert set(actual) == set(expected)
    for symbol in symbols:
        np.testing.assert_array_equal(actual[symbol], expected[symbol])
    assert calls["grid"] == 1
    assert calls["events_many"] == 1
    assert calls["get_coverage"] == 0
    assert calls["query_events"] == 0

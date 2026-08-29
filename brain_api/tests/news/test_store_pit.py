from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

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


def _window() -> NewsWindow:
    end = datetime(2026, 8, 24, 9, 0, tzinfo=NY)
    start = datetime(2026, 8, 17, 9, 0, tzinfo=NY)
    return NewsWindow(start_exclusive=start, end_inclusive=end)


def _event(
    article_id: str,
    *,
    created: datetime,
    updated: datetime,
    score: float,
    symbol: str = "AAPL",
) -> NewsEvent:
    now = utcnow()
    return NewsEvent(
        provider=NEWS_PROVIDER,
        provider_article_id=article_id,
        symbol=symbol,
        created_at=created,
        updated_at=updated,
        source="benzinga",
        sentiment_score=score,
        p_positive=0.6,
        p_negative=0.1,
        p_neutral=0.3,
        confidence=0.6,
        scored_text_sha256="abc",
        sentiment_model=NEWS_SENTIMENT_MODEL,
        sentiment_model_revision=NEWS_SENTIMENT_REVISION,
        schema_version=NEWS_SCHEMA_VERSION,
        ingested_at=now,
    )


def test_query_uses_latest_revision_at_or_before_cutoff(tmp_path) -> None:
    store = NewsStore(tmp_path)
    window = _window()
    created = window.start_exclusive + timedelta(days=1)
    old = _event("art-1", created=created, updated=created, score=0.1)
    new = _event(
        "art-1",
        created=created,
        updated=window.end_inclusive - timedelta(hours=1),
        score=0.9,
    )
    future = _event(
        "art-1",
        created=created,
        updated=window.end_inclusive + timedelta(hours=1),
        score=-0.5,
    )
    coverage = NewsCoverage(
        provider=NEWS_PROVIDER,
        symbol="AAPL",
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
    store.commit_window(events=[old, new, future], coverage=coverage, cache_rows=[])
    events = store.query_events(["AAPL"], window)
    assert len(events) == 1
    assert events[0].sentiment_score == 0.9


def test_query_events_many_matches_point_in_time_weekly_reads(tmp_path) -> None:
    store = NewsStore(tmp_path)
    first = _window()
    second = NewsWindow(
        start_exclusive=first.end_inclusive,
        end_inclusive=first.end_inclusive + timedelta(days=7),
    )
    first_created = first.start_exclusive + timedelta(days=1)
    second_created = second.start_exclusive + timedelta(days=1)
    rows = [
        _event("first", created=first_created, updated=first_created, score=0.1),
        _event(
            "first",
            created=first_created,
            updated=first.end_inclusive - timedelta(hours=1),
            score=0.8,
        ),
        _event(
            "first",
            created=first_created,
            updated=first.end_inclusive + timedelta(hours=1),
            score=-0.5,
        ),
        _event("second", created=second_created, updated=second_created, score=0.3),
    ]
    coverage = NewsCoverage(
        provider=NEWS_PROVIDER,
        symbol="AAPL",
        window_start_exclusive=first.start_exclusive,
        window_end_inclusive=first.end_inclusive,
        schema_version=NEWS_SCHEMA_VERSION,
        sentiment_model=NEWS_SENTIMENT_MODEL,
        sentiment_model_revision=NEWS_SENTIMENT_REVISION,
        status="complete",
        page_count=1,
        event_count=1,
        future_revision_excluded_count=1,
        fetched_at=utcnow(),
        request_manifest_hash="m",
    )
    store.commit_window(events=rows, coverage=coverage, cache_rows=[])

    bulk = store.query_events_many(["AAPL"], [first, second])

    assert bulk[first] == store.query_events(["AAPL"], first)
    assert bulk[second] == store.query_events(["AAPL"], second)
    assert [event.sentiment_score for event in bulk[first]] == [0.8]
    assert [event.sentiment_score for event in bulk[second]] == [0.3]


def test_query_drops_created_at_on_exclusive_start(tmp_path) -> None:
    store = NewsStore(tmp_path)
    window = _window()
    boundary = _event(
        "edge",
        created=window.start_exclusive,
        updated=window.start_exclusive,
        score=0.2,
    )
    inside = _event(
        "in",
        created=window.start_exclusive + timedelta(seconds=1),
        updated=window.start_exclusive + timedelta(seconds=1),
        score=0.3,
    )
    coverage = NewsCoverage(
        provider=NEWS_PROVIDER,
        symbol="AAPL",
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
    store.commit_window(events=[boundary, inside], coverage=coverage, cache_rows=[])
    events = store.query_events(["AAPL"], window)
    assert [event.provider_article_id for event in events] == ["in"]


def test_coverage_is_exact_window_not_superset(tmp_path) -> None:
    store = NewsStore(tmp_path)
    window = _window()
    wide = NewsWindow(
        start_exclusive=window.start_exclusive - timedelta(days=7),
        end_inclusive=window.end_inclusive,
    )
    coverage = NewsCoverage(
        provider=NEWS_PROVIDER,
        symbol="AAPL",
        window_start_exclusive=wide.start_exclusive,
        window_end_inclusive=wide.end_inclusive,
        schema_version=NEWS_SCHEMA_VERSION,
        sentiment_model=NEWS_SENTIMENT_MODEL,
        sentiment_model_revision=NEWS_SENTIMENT_REVISION,
        status="complete",
        page_count=1,
        event_count=0,
        future_revision_excluded_count=0,
        fetched_at=utcnow(),
        request_manifest_hash="m",
    )
    store.commit_window(events=[], coverage=coverage, cache_rows=[])
    assert store.get_coverage("AAPL", window) is None
    assert store.get_coverage("AAPL", wide) is not None


def test_finbert_sha_is_part_of_coverage_identity(tmp_path) -> None:
    store = NewsStore(tmp_path)
    window = _window()
    coverage = NewsCoverage(
        provider=NEWS_PROVIDER,
        symbol="AAPL",
        window_start_exclusive=window.start_exclusive,
        window_end_inclusive=window.end_inclusive,
        schema_version=NEWS_SCHEMA_VERSION,
        sentiment_model=NEWS_SENTIMENT_MODEL,
        sentiment_model_revision="deadbeef",
        status="complete",
        page_count=1,
        event_count=0,
        future_revision_excluded_count=0,
        fetched_at=utcnow(),
        request_manifest_hash="m",
    )
    store.commit_window(events=[], coverage=coverage, cache_rows=[])
    assert store.get_coverage("AAPL", window) is None

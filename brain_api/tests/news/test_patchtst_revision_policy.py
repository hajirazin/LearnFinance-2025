"""Historical revision approval does not alter publication membership or live queries."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from brain_api.news.models import (
    NEWS_PROVIDER,
    NEWS_SCHEMA_VERSION,
    NEWS_SENTIMENT_MODEL,
    NEWS_SENTIMENT_REVISION,
    NewsCoverage,
    NewsEvent,
)
from brain_api.news.patchtst import load_session_sentiment, session_news_window
from brain_api.news.store import NewsStore
from brain_api.news_etl.windows import missing_session_windows


@pytest.fixture
def revision_case(tmp_path):
    store = NewsStore(tmp_path)
    end = datetime(2021, 3, 22, 20, tzinfo=UTC)
    window = session_news_window(end)
    coverage = NewsCoverage(
        NEWS_PROVIDER,
        "AAA",
        window.start_exclusive,
        end,
        NEWS_SCHEMA_VERSION,
        NEWS_SENTIMENT_MODEL,
        NEWS_SENTIMENT_REVISION,
        "verified_empty",
        1,
        0,
        0,
        end + timedelta(days=3),
        "test-proof",
    )
    event = NewsEvent(
        NEWS_PROVIDER,
        "article-1",
        "AAA",
        end - timedelta(hours=1),
        end + timedelta(days=2),
        "benzinga",
        0.8,
        0.9,
        0.05,
        0.05,
        0.9,
        "test-hash",
        NEWS_SENTIMENT_MODEL,
        NEWS_SENTIMENT_REVISION,
        NEWS_SCHEMA_VERSION,
        end + timedelta(days=3),
    )
    grid = pd.Series([end], index=pd.DatetimeIndex(["2021-03-22"]))
    return store, coverage, event, grid


def test_approved_later_revision_fills_the_known_historical_gap(revision_case):
    store, coverage, event, grid = revision_case
    store.commit_window(events=[event], coverage=coverage, cache_rows=[])
    strict = load_session_sentiment(store, ["AAA"], grid)
    assert pd.isna(strict.iloc[0, 0])
    approved = load_session_sentiment(
        store, ["AAA"], grid, revision_policy="allow_later_revisions"
    )
    assert approved.iloc[0, 0] == pytest.approx(0.8)
    assert approved.attrs["later_revision_cells"] == 1
    assert approved.attrs["revision_policy"] == "allow_later_revisions"
    assert store.query_events(["AAA"], session_news_window(grid.iloc[0])) == []
    assert not missing_session_windows(
        store, ["AAA"], [session_news_window(grid.iloc[0])]
    )


def test_earlier_available_revision_is_preferred_even_when_later_is_allowed(
    revision_case,
):
    store, coverage, event, grid = revision_case
    old = replace(event, updated_at=event.created_at, sentiment_score=-0.4)
    store.commit_window(events=[old, event], coverage=coverage, cache_rows=[])
    approved = load_session_sentiment(
        store, ["AAA"], grid, revision_policy="allow_later_revisions"
    )
    assert approved.iloc[0, 0] == pytest.approx(-0.4)
    assert approved.attrs["later_revision_cells"] == 0


def test_articles_published_after_cutoff_are_still_excluded(revision_case):
    store, coverage, event, grid = revision_case
    event = replace(event, created_at=grid.iloc[0] + timedelta(hours=1))
    store.commit_window(events=[event], coverage=coverage, cache_rows=[])
    approved = load_session_sentiment(
        store, ["AAA"], grid, revision_policy="allow_later_revisions"
    )
    assert approved.iloc[0, 0] == 0
    assert approved.attrs["later_revision_cells"] == 0


def test_approval_does_not_convert_missing_coverage_to_zero(revision_case):
    store, coverage, event, grid = revision_case
    store.commit_window(events=[event], coverage=coverage, cache_rows=[])
    approved = load_session_sentiment(
        store, ["MISSING"], grid, revision_policy="allow_later_revisions"
    )
    assert pd.isna(approved.iloc[0, 0])


def test_unstored_excluded_revisions_still_require_evidence(revision_case):
    store, coverage, _, grid = revision_case
    store.commit_window(
        events=[],
        coverage=replace(coverage, future_revision_excluded_count=1),
        cache_rows=[],
    )
    approved = load_session_sentiment(
        store, ["AAA"], grid, revision_policy="allow_later_revisions"
    )
    assert pd.isna(approved.iloc[0, 0])


def test_unknown_revision_policy_is_not_silently_accepted(revision_case):
    store, _, _, grid = revision_case
    with pytest.raises(ValueError, match="revision policy"):
        load_session_sentiment(store, ["AAA"], grid, revision_policy="unknown")

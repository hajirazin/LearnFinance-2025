"""Deterministic correctness tests for point-in-time SAC news inputs."""

import math
from datetime import date

import pytest

from brain_api.core.news_sentiment.fetcher import NewsProviderError, YFinanceNewsFetcher
from brain_api.core.news_sentiment.weekly import (
    DailyNewsObservation,
    NewsObservationError,
    aggregate_weekly_news_observation,
)


def test_weekly_news_uses_article_confidence_and_recency_weights():
    as_of = date(2026, 5, 4)
    rows = [
        DailyNewsObservation(date(2026, 5, 4), 0.8, 2, 0.5),
        DailyNewsObservation(date(2026, 4, 27), -0.4, 1, 1.0),
    ]

    result = aggregate_weekly_news_observation(rows, as_of)

    recent_weight = 2 * 0.5
    old_weight = math.exp(-1)
    expected = (0.8 * recent_weight - 0.4 * old_weight) / (recent_weight + old_weight)
    assert result.sentiment_score == pytest.approx(expected)
    assert result.coverage == 1.0
    assert result.article_count == 3
    assert not result.confirmed_no_articles


def test_confirmed_zero_news_is_neutral_but_missing_window_fails():
    as_of = date(2026, 5, 4)
    result = aggregate_weekly_news_observation(
        [DailyNewsObservation(as_of, 0.0, 0, 0.0)],
        as_of,
    )

    assert result.sentiment_score == 0.0
    assert result.coverage == 0.0
    assert result.confirmed_no_articles
    with pytest.raises(NewsObservationError, match="No provider-checked"):
        aggregate_weekly_news_observation([], as_of)


def test_news_provider_error_is_not_converted_to_confirmed_zero(monkeypatch):
    class BrokenTicker:
        @property
        def news(self):
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(
        "brain_api.core.news_sentiment.fetcher.yf.Ticker",
        lambda _symbol: BrokenTicker(),
    )

    with pytest.raises(NewsProviderError, match="provider unavailable"):
        YFinanceNewsFetcher().fetch("AAPL", 10)

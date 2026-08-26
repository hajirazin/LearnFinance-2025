from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from brain_api.core.ppo_discovery.news_adapter import (
    build_ppo_news_features,
    load_weekly_ppo_news_features,
)
from brain_api.core.sac.news_adapter import build_sac_news_features
from brain_api.news.models import (
    NEWS_PROVIDER,
    NEWS_SCHEMA_VERSION,
    NEWS_SENTIMENT_MODEL,
    NEWS_SENTIMENT_REVISION,
    NewsEvent,
)
from brain_api.news.store import utcnow

NY = ZoneInfo("America/New_York")
CUTOFF = datetime(2026, 8, 24, 9, 0, tzinfo=NY)


def _event(symbol: str, score: float, hours_ago: float, article_id: str) -> NewsEvent:
    created = CUTOFF - timedelta(hours=hours_ago)
    return NewsEvent(
        provider=NEWS_PROVIDER,
        provider_article_id=article_id,
        symbol=symbol,
        created_at=created,
        updated_at=created,
        source="benzinga",
        sentiment_score=score,
        p_positive=max(score, 0),
        p_negative=max(-score, 0),
        p_neutral=0.0,
        confidence=1.0,
        scored_text_sha256="x",
        sentiment_model=NEWS_SENTIMENT_MODEL,
        sentiment_model_revision=NEWS_SENTIMENT_REVISION,
        schema_version=NEWS_SCHEMA_VERSION,
        ingested_at=utcnow(),
    )


def test_sac_empty_is_zero() -> None:
    assert build_sac_news_features({"AAPL": []}, cutoff=CUTOFF) == {"AAPL": 0.0}


def test_sac_one_article() -> None:
    features = build_sac_news_features(
        {"AAPL": [_event("AAPL", 0.4, 1, "1")]}, cutoff=CUTOFF
    )
    assert features["AAPL"] == pytest.approx(0.4)


def test_ppo_dispersion_and_count() -> None:
    events = [
        _event("AAPL", 0.0, 1, "1"),
        _event("AAPL", 2.0, 2, "2"),
    ]
    features = build_ppo_news_features({"AAPL": events}, cutoff=CUTOFF)
    row = features["AAPL"]
    assert row.article_count == 2
    assert row.log1p_article_count == pytest.approx(math.log1p(2))
    assert row.sentiment_dispersion == pytest.approx(1.0)
    assert row.raw_sentiment != 0.0
    assert 0.0 < row.recency <= 1.0


def test_friday_actor_cutoff_uses_following_monday_window() -> None:
    friday = datetime(2026, 8, 21, 20, 0, tzinfo=UTC)
    captured: dict = {}

    class _FakeStore:
        def require_coverage(self, symbols, window):
            captured["window"] = window
            return {}

        def query_events(self, symbols, window):
            return []

    rows = load_weekly_ppo_news_features(friday, ["AAPL"], store=_FakeStore())
    window = captured["window"]
    assert window.start_exclusive == datetime(2026, 8, 17, 9, 0, tzinfo=NY)
    assert window.end_inclusive == datetime(2026, 8, 24, 9, 0, tzinfo=NY)
    assert rows["AAPL"].article_count == 0

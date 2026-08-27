from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from brain_api.news.alpaca_provider import WindowBatchFetch
from brain_api.news.errors import NewsCapExceeded
from brain_api.news.models import NewsWindow, ProviderArticle
from brain_api.news.sentiment import ScoredSentiment, scored_text_sha256
from brain_api.news.service import NewsService
from brain_api.news.store import NewsStore

NY = ZoneInfo("America/New_York")


def _window() -> NewsWindow:
    return NewsWindow(
        start_exclusive=datetime(2026, 8, 17, 9, 0, tzinfo=NY),
        end_inclusive=datetime(2026, 8, 24, 9, 0, tzinfo=NY),
    )


def _article(symbol: str, article_id: str = "1") -> ProviderArticle:
    created = datetime(2026, 8, 18, 9, 0, tzinfo=NY)
    return ProviderArticle(
        provider_article_id=article_id,
        symbol=symbol,
        created_at=created,
        updated_at=created,
        source="benzinga",
        headline="Hello",
        summary="World",
    )


class FakeScorer:
    batch_size = 32

    def score_texts(self, texts: list[str]) -> list[ScoredSentiment]:
        return [
            ScoredSentiment(
                sentiment_score=0.2,
                p_positive=0.6,
                p_negative=0.4,
                p_neutral=0.0,
                confidence=0.6,
                scored_text_sha256=scored_text_sha256(text),
                scored_text=text,
            )
            for text in texts
        ]


class _BatchProvider:
    def __init__(
        self,
        result: WindowBatchFetch,
        *,
        fallback: dict[str, list[ProviderArticle]] | None = None,
        fallback_error: Exception | None = None,
    ) -> None:
        self.result = result
        self.fallback = fallback or {}
        self.fallback_error = fallback_error
        self.batch_calls: list[tuple[str, ...]] = []
        self.window_calls: list[str] = []

    def fetch_window_batch(self, symbols, window) -> WindowBatchFetch:
        self.batch_calls.append(tuple(symbols))
        return self.result

    def fetch_window(self, symbol: str, window):
        self.window_calls.append(symbol)
        if self.fallback_error is not None:
            raise self.fallback_error
        return self.fallback.get(symbol, []), 1


def test_unproven_miss_does_not_write_verified_empty_before_fallback(tmp_path) -> None:
    window = _window()
    store = NewsStore(tmp_path)
    provider = _BatchProvider(
        WindowBatchFetch(
            articles_by_symbol={"AAPL": [_article("AAPL")], "MSFT": []},
            page_count=1,
            empties_are_proven=False,
        ),
        fallback_error=NewsCapExceeded("MSFT leftover"),
    )
    service = NewsService(store, provider=provider, scorer=FakeScorer())
    with pytest.raises(NewsCapExceeded, match="leftover"):
        service.materialize(["AAPL", "MSFT"], window)
    assert store.get_coverage("AAPL", window) is not None
    assert store.get_coverage("AAPL", window).status == "complete"
    assert store.get_coverage("MSFT", window) is None
    assert provider.window_calls == ["MSFT"]


def test_proven_empties_write_verified_empty_without_fallback(tmp_path) -> None:
    window = _window()
    store = NewsStore(tmp_path)
    provider = _BatchProvider(
        WindowBatchFetch(
            articles_by_symbol={"AAPL": [_article("AAPL")], "MSFT": []},
            page_count=1,
            empties_are_proven=True,
        )
    )
    service = NewsService(store, provider=provider, scorer=FakeScorer())
    coverage, events = service.materialize(["AAPL", "MSFT"], window)
    statuses = {row.symbol: row.status for row in coverage}
    assert statuses == {"AAPL": "complete", "MSFT": "verified_empty"}
    assert provider.window_calls == []
    assert len(events) == 1


def test_unproven_miss_falls_back_to_single_ticker(tmp_path) -> None:
    window = _window()
    store = NewsStore(tmp_path)
    provider = _BatchProvider(
        WindowBatchFetch(
            articles_by_symbol={"AAPL": [_article("AAPL")], "MSFT": []},
            page_count=1,
            empties_are_proven=False,
        ),
        fallback={"MSFT": [_article("MSFT", "msft-1")]},
    )
    service = NewsService(store, provider=provider, scorer=FakeScorer())
    coverage, events = service.materialize(["AAPL", "MSFT"], window)
    statuses = {row.symbol: row.status for row in coverage}
    assert statuses == {"AAPL": "complete", "MSFT": "complete"}
    assert provider.window_calls == ["MSFT"]
    assert {event.symbol for event in events} == {"AAPL", "MSFT"}


def test_provider_without_batch_still_materializes(tmp_path) -> None:
    window = _window()
    store = NewsStore(tmp_path)

    class SingleOnly:
        def fetch_window(self, symbol: str, window):
            created = window.start_exclusive + timedelta(days=1)
            article = ProviderArticle(
                provider_article_id=f"{symbol}-1",
                symbol=symbol,
                created_at=created,
                updated_at=created,
                source="benzinga",
                headline="Hello",
                summary="World",
            )
            return [article], 1

    service = NewsService(store, provider=SingleOnly(), scorer=FakeScorer())
    coverage, events = service.materialize(["AAPL"], window)
    assert coverage[0].status == "complete"
    assert len(events) == 1

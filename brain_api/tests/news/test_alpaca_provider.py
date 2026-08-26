from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from brain_api.news.alpaca_provider import AlpacaNewsProvider
from brain_api.news.errors import RepeatedPageTokenError
from brain_api.news.models import NewsWindow

NY = ZoneInfo("America/New_York")


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


class _FakeSession:
    def __init__(self, pages: list[tuple[int, dict]]) -> None:
        self.pages = list(pages)
        self.calls: list[dict] = []

    def get(self, url, headers, params, timeout):
        self.calls.append(params)
        status, payload = self.pages.pop(0)
        return _FakeResponse(status, payload)


def _window() -> NewsWindow:
    return NewsWindow(
        start_exclusive=datetime(2026, 8, 17, 9, 0, tzinfo=NY),
        end_inclusive=datetime(2026, 8, 24, 9, 0, tzinfo=NY),
    )


def test_local_filter_drops_created_at_on_start(monkeypatch) -> None:
    window = _window()
    payload = {
        "news": [
            {
                "id": "on-start",
                "headline": "A",
                "summary": "",
                "created_at": window.start_exclusive.isoformat(),
                "updated_at": window.start_exclusive.isoformat(),
                "source": "benzinga",
            },
            {
                "id": "inside",
                "headline": "B",
                "summary": "",
                "created_at": datetime(2026, 8, 18, 9, 0, tzinfo=NY).isoformat(),
                "updated_at": datetime(2026, 8, 18, 9, 0, tzinfo=NY).isoformat(),
                "source": "benzinga",
            },
        ]
    }
    session = _FakeSession([(200, payload)])
    provider = AlpacaNewsProvider(
        api_key="k", api_secret="s", session=session, rate_limit_delay=0
    )
    articles, pages = provider.fetch_window("AAPL", window)
    assert pages == 1
    assert [article.provider_article_id for article in articles] == ["inside"]
    assert "start" in session.calls[0]
    assert "end" in session.calls[0]


def test_repeated_page_token_aborts() -> None:
    window = _window()
    article = {
        "id": "1",
        "headline": "A",
        "summary": "",
        "created_at": datetime(2026, 8, 18, 9, 0, tzinfo=NY).isoformat(),
        "updated_at": datetime(2026, 8, 18, 9, 0, tzinfo=NY).isoformat(),
        "source": "benzinga",
    }
    session = _FakeSession(
        [
            (200, {"news": [article], "next_page_token": "tok"}),
            (200, {"news": [article], "next_page_token": "tok"}),
        ]
    )
    provider = AlpacaNewsProvider(
        api_key="k", api_secret="s", session=session, rate_limit_delay=0
    )
    with pytest.raises(RepeatedPageTokenError):
        provider.fetch_window("AAPL", window)


def test_retries_429_then_succeeds() -> None:
    window = _window()
    article = {
        "id": "1",
        "headline": "A",
        "summary": "",
        "created_at": datetime(2026, 8, 18, 9, 0, tzinfo=NY).isoformat(),
        "updated_at": datetime(2026, 8, 18, 9, 0, tzinfo=NY).isoformat(),
        "source": "benzinga",
    }
    session = _FakeSession(
        [
            (429, {"news": []}),
            (200, {"news": [article]}),
        ]
    )
    provider = AlpacaNewsProvider(
        api_key="k", api_secret="s", session=session, rate_limit_delay=0
    )
    articles, pages = provider.fetch_window("AAPL", window)
    assert pages == 1
    assert articles[0].provider_article_id == "1"


def test_article_cap_aborts_during_pagination(monkeypatch) -> None:
    from brain_api.news import alpaca_provider as provider_mod
    from brain_api.news.errors import NewsCapExceeded

    monkeypatch.setattr(provider_mod, "MAX_ARTICLES_PER_SYMBOL_WINDOW", 2)
    window = _window()

    def _article(article_id: str) -> dict:
        return {
            "id": article_id,
            "headline": "A",
            "summary": "",
            "created_at": datetime(2026, 8, 18, 9, 0, tzinfo=NY).isoformat(),
            "updated_at": datetime(2026, 8, 18, 9, 0, tzinfo=NY).isoformat(),
            "source": "benzinga",
        }

    session = _FakeSession(
        [
            (200, {"news": [_article("1"), _article("2")], "next_page_token": "p2"}),
            (200, {"news": [_article("3")], "next_page_token": "p3"}),
            (200, {"news": [_article("4")]}),
        ]
    )
    provider = AlpacaNewsProvider(
        api_key="k", api_secret="s", session=session, rate_limit_delay=0
    )
    with pytest.raises(NewsCapExceeded):
        provider.fetch_window("AAPL", window)
    assert len(session.calls) == 2


def test_naive_provider_timestamp_is_rejected() -> None:
    from brain_api.news.errors import NewsProviderError

    window = _window()
    payload = {
        "news": [
            {
                "id": "naive",
                "headline": "A",
                "summary": "",
                "created_at": "2026-08-18T09:00:00",
                "updated_at": "2026-08-18T09:00:00",
                "source": "benzinga",
            }
        ]
    }
    session = _FakeSession([(200, payload)])
    provider = AlpacaNewsProvider(
        api_key="k", api_secret="s", session=session, rate_limit_delay=0
    )
    with pytest.raises(NewsProviderError, match="timezone-naive"):
        provider.fetch_window("AAPL", window)

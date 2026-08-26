from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient

from brain_api.main import app
from brain_api.news.models import NewsWindow, ProviderArticle
from brain_api.news.sentiment import ScoredSentiment, scored_text_sha256
from brain_api.news.service import NewsService
from brain_api.news.store import NewsStore
from brain_api.routes.news import endpoints as news_endpoints

NY = ZoneInfo("America/New_York")
START = datetime(2026, 8, 17, 9, 0, tzinfo=NY)
END = datetime(2026, 8, 24, 9, 0, tzinfo=NY)


class _Provider:
    def fetch_window(self, symbol: str, window: NewsWindow):
        created = window.start_exclusive + timedelta(days=1)
        return [
            ProviderArticle(
                provider_article_id=f"{symbol}-1",
                symbol=symbol,
                created_at=created,
                updated_at=created,
                source="benzinga",
                headline="Hello",
                summary="World",
            )
        ], 1


class _Scorer:
    batch_size = 32

    def score_texts(self, texts: list[str]) -> list[ScoredSentiment]:
        return [
            ScoredSentiment(0.1, 0.5, 0.4, 0.1, 0.5, scored_text_sha256(text), text)
            for text in texts
        ]


def test_materialize_and_query_and_calendar(tmp_path, monkeypatch, caplog) -> None:
    store = NewsStore(tmp_path)
    service = NewsService(store, provider=_Provider(), scorer=_Scorer())
    monkeypatch.setattr(news_endpoints, "get_news_service", lambda: service)
    client = TestClient(app)
    import logging

    caplog.set_level(logging.INFO)
    body = {
        "symbols": ["AAPL", "MSFT", "GOOG"],
        "start_exclusive": START.isoformat(),
        "end_inclusive": END.isoformat(),
    }
    response = client.post("/news/windows/materialize", json=body)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert len(payload["coverage"]) == 3
    assert len(payload["events"]) == 3
    query = client.post("/news/windows/query", json=body)
    assert query.status_code == 200
    symbol_lines = [
        record.message for record in caplog.records if "news symbol=" in record.message
    ]
    assert len(symbol_lines) == 3
    assert not any("article_id" in record.message.lower() for record in caplog.records)

    missing = client.post(
        "/news/windows/query",
        json={**body, "symbols": ["AAPL", "TSLA"]},
    )
    assert missing.status_code == 422

    calendar = client.post(
        "/calendar/monday-decision-window",
        json={"as_of": END.isoformat()},
    )
    assert calendar.status_code == 200
    assert calendar.json()["cutoff"] == END.isoformat()
    bad = client.post(
        "/calendar/monday-decision-window",
        json={"as_of": datetime(2026, 8, 24, 10, 0, tzinfo=NY).isoformat()},
    )
    assert bad.status_code == 422


def test_materialize_rejects_future_window_end(tmp_path, monkeypatch) -> None:
    store = NewsStore(tmp_path)
    service = NewsService(store, provider=_Provider(), scorer=_Scorer())
    monkeypatch.setattr(news_endpoints, "get_news_service", lambda: service)
    client = TestClient(app)
    now = datetime.now(tz=NY)
    future_end = now + timedelta(days=7)
    response = client.post(
        "/news/windows/materialize",
        json={
            "symbols": ["AAPL"],
            "start_exclusive": (future_end - timedelta(days=7)).isoformat(),
            "end_inclusive": future_end.isoformat(),
        },
    )
    assert response.status_code == 422
    assert "future" in response.json()["detail"].lower()


def test_legacy_signals_news_is_gone() -> None:
    client = TestClient(app)
    response = client.post("/signals/news", json={"symbols": ["AAPL"]})
    assert response.status_code == 404

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from brain_api.news.models import NewsWindow, ProviderArticle
from brain_api.news.sentiment import ScoredSentiment, scored_text_sha256
from brain_api.news.service import NewsService
from brain_api.news.store import NewsStore

NY = ZoneInfo("America/New_York")


class FakeProvider:
    def __init__(self) -> None:
        self.calls = 0
        self.gate = threading.Event()
        self.started = threading.Event()

    def fetch_window(self, symbol: str, window: NewsWindow):
        self.started.set()
        self.gate.wait(timeout=2)
        self.calls += 1
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


class FakeScorer:
    batch_size = 32
    calls = 0

    def score_texts(self, texts: list[str]) -> list[ScoredSentiment]:
        type(self).calls += len(texts)
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


def test_overlapping_materialize_scores_once(tmp_path) -> None:
    FakeScorer.calls = 0
    window = NewsWindow(
        start_exclusive=datetime(2026, 8, 17, 9, 0, tzinfo=NY),
        end_inclusive=datetime(2026, 8, 24, 9, 0, tzinfo=NY),
    )
    store = NewsStore(tmp_path)
    provider = FakeProvider()
    service = NewsService(store, provider=provider, scorer=FakeScorer())
    errors: list[BaseException] = []

    def run() -> None:
        try:
            service.materialize(["AAPL"], window)
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=run)
    second = threading.Thread(target=run)
    first.start()
    assert provider.started.wait(timeout=2)
    second.start()
    time.sleep(0.05)
    provider.gate.set()
    first.join(timeout=5)
    second.join(timeout=5)
    assert errors == []
    assert FakeScorer.calls == 1
    coverage, events = service.query(["AAPL"], window)
    assert coverage[0].status == "complete"
    assert len(events) == 1

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import Event
from zoneinfo import ZoneInfo

from brain_api.news.models import (
    NEWS_PROVIDER,
    NEWS_SCHEMA_VERSION,
    NEWS_SENTIMENT_MODEL,
    NEWS_SENTIMENT_REVISION,
    NewsCoverage,
    NewsWindow,
)
from brain_api.news.store import NewsStore, utcnow
from brain_api.news_etl.backfill import run_backfill
from brain_api.news_etl.gaps import missing_windows
from brain_api.news_etl.jobs import job_windows_total

NY = ZoneInfo("America/New_York")
START = datetime(2026, 8, 17, 9, 0, tzinfo=NY)
END = datetime(2026, 8, 24, 9, 0, tzinfo=NY)
RANGE_START = datetime(2026, 8, 10, 9, 0, tzinfo=NY)
RANGE_END = datetime(2026, 8, 24, 9, 0, tzinfo=NY)
WINDOW_ONE = NewsWindow(
    start_exclusive=datetime(2026, 8, 10, 9, 0, tzinfo=NY),
    end_inclusive=datetime(2026, 8, 17, 9, 0, tzinfo=NY),
)
WINDOW_TWO = NewsWindow(
    start_exclusive=datetime(2026, 8, 17, 9, 0, tzinfo=NY),
    end_inclusive=datetime(2026, 8, 24, 9, 0, tzinfo=NY),
)


class _MustNotMaterialize:
    def materialize(self, symbols, window):
        raise AssertionError(f"unexpected materialize symbols={symbols}")


class _RecordingService:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], NewsWindow]] = []

    def materialize(self, symbols, window):
        self.calls.append((list(symbols), window))
        return [], []


def _coverage(symbol: str, window: NewsWindow) -> NewsCoverage:
    return NewsCoverage(
        provider=NEWS_PROVIDER,
        symbol=symbol,
        window_start_exclusive=window.start_exclusive,
        window_end_inclusive=window.end_inclusive,
        schema_version=NEWS_SCHEMA_VERSION,
        sentiment_model=NEWS_SENTIMENT_MODEL,
        sentiment_model_revision=NEWS_SENTIMENT_REVISION,
        status="verified_empty",
        page_count=1,
        event_count=0,
        future_revision_excluded_count=0,
        fetched_at=utcnow(),
        request_manifest_hash="m",
    )


def test_backfill_logs_skip_summary_when_all_covered(tmp_path, caplog) -> None:
    store = NewsStore(tmp_path)
    store.commit_window(
        events=[], coverage=_coverage("AAPL", WINDOW_TWO), cache_rows=[]
    )
    caplog.set_level(logging.INFO)
    job_id = run_backfill(
        symbols=["AAPL"],
        start=START,
        end=END,
        store=store,
        service=_MustNotMaterialize(),
    )
    job = store.get_job(job_id)
    assert job.status == "complete"
    assert job.windows_done == job.windows_total
    assert any("skip-summary" in record.message for record in caplog.records)


def test_backfill_materializes_only_uncovered_cells(tmp_path) -> None:
    store = NewsStore(tmp_path)
    store.commit_window(
        events=[], coverage=_coverage("AAPL", WINDOW_ONE), cache_rows=[]
    )
    store.commit_window(
        events=[], coverage=_coverage("AAPL", WINDOW_TWO), cache_rows=[]
    )
    store.commit_window(
        events=[], coverage=_coverage("MSFT", WINDOW_ONE), cache_rows=[]
    )
    service = _RecordingService()
    job_id = run_backfill(
        symbols=["AAPL", "MSFT"],
        start=RANGE_START,
        end=RANGE_END,
        store=store,
        service=service,
    )
    job = store.get_job(job_id)
    assert job.status == "complete"
    assert job.windows_done == 4
    assert job.windows_total == job_windows_total(
        ["AAPL", "MSFT"], start=RANGE_START, end=RANGE_END
    )
    assert len(service.calls) == 1
    symbols, window = service.calls[0]
    assert symbols == ["MSFT"]
    assert window.start_exclusive == WINDOW_TWO.start_exclusive
    assert window.end_inclusive == WINDOW_TWO.end_inclusive


def test_backfill_all_covered_is_complete_without_materialize(tmp_path) -> None:
    store = NewsStore(tmp_path)
    for window in (WINDOW_ONE, WINDOW_TWO):
        for symbol in ("AAPL", "MSFT"):
            store.commit_window(
                events=[], coverage=_coverage(symbol, window), cache_rows=[]
            )
    job_id = run_backfill(
        symbols=["AAPL", "MSFT"],
        start=RANGE_START,
        end=RANGE_END,
        store=store,
        service=_MustNotMaterialize(),
    )
    job = store.get_job(job_id)
    assert job.status == "complete"
    assert job.windows_done == job.windows_total == 4


def test_missing_windows_returns_only_uncovered_pairs(tmp_path) -> None:
    store = NewsStore(tmp_path)
    store.commit_window(
        events=[], coverage=_coverage("AAPL", WINDOW_ONE), cache_rows=[]
    )
    store.commit_window(
        events=[], coverage=_coverage("AAPL", WINDOW_TWO), cache_rows=[]
    )
    store.commit_window(
        events=[], coverage=_coverage("MSFT", WINDOW_ONE), cache_rows=[]
    )
    gaps = missing_windows(
        store, symbols=["AAPL", "MSFT"], start=RANGE_START, end=RANGE_END
    )
    assert [(symbol, window.end_inclusive) for symbol, window in gaps] == [
        ("MSFT", WINDOW_TWO.end_inclusive)
    ]


def test_concurrent_backfill_fetches_and_publishes_once(tmp_path, monkeypatch) -> None:
    store = NewsStore(tmp_path)
    entered, release = Event(), Event()
    calls, sync_calls = [], []

    class BlockingService:
        def materialize(self, symbols, window):
            calls.append(list(symbols))
            assert len(calls) == 1, "duplicate backfill started"
            entered.set()
            assert release.wait(timeout=10)
            return [], []

    def synchronize(store, *, publish=False):
        sync_calls.append(publish)

    monkeypatch.setattr("brain_api.news_etl.backfill.sync_news", synchronize)
    kwargs = {
        "symbols": ["AAPL"],
        "start": START,
        "end": END,
        "store": store,
        "service": BlockingService(),
        "shared": True,
    }
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(run_backfill, **kwargs)
        assert entered.wait(timeout=10)
        try:
            duplicate = run_backfill(**kwargs)
        finally:
            release.set()
        assert first.result(timeout=10) == duplicate
    assert calls == [["AAPL"]]
    assert sync_calls == [False, True]
    assert store.get_job(duplicate).status == "complete"


def test_backfill_releases_job_lock_after_failure(tmp_path) -> None:
    import pytest

    store = NewsStore(tmp_path)
    kwargs = {"symbols": ["AAPL"], "start": START, "end": END, "store": store}
    with pytest.raises(AssertionError, match="unexpected materialize"):
        run_backfill(**kwargs, service=_MustNotMaterialize())
    job_id = run_backfill(**kwargs, service=_RecordingService())
    assert store.get_job(job_id).status == "complete"

"""The shared news data restores coverage and scores on a fresh machine."""

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from huggingface_hub.errors import EntryNotFoundError

from brain_api.news import sync
from brain_api.news.models import (
    NEWS_PROVIDER,
    NEWS_SCHEMA_VERSION,
    NEWS_SENTIMENT_MODEL,
    NEWS_SENTIMENT_REVISION,
    NewsCoverage,
    NewsWindow,
)
from brain_api.news.store import NewsStore
from brain_api.news.sync_partitions import export_partition, import_partition
from brain_api.news_etl.backfill import run_backfill


@pytest.fixture
def shared_hub(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_NEWS_SENTIMENT_REPO", "test/shared-news")

    class Hub:
        def __init__(self):
            self.sha = "initial"
            self.tree = {}

        def create_repo(self, *args, **kwargs):
            assert kwargs["private"]

        def repo_info(self, *args, **kwargs):
            return SimpleNamespace(private=True, sha=self.sha)

        def create_commit(self, repo, *, operations, parent_commit, **kwargs):
            assert parent_commit == self.sha
            for operation in operations:
                data = operation.path_or_fileobj
                self.tree[operation.path_in_repo] = (
                    data if isinstance(data, bytes) else Path(data).read_bytes()
                )
            self.sha = str(len(self.tree))
            return SimpleNamespace(oid=self.sha)

    hub = Hub()

    def download(repo, revision, name, token):
        if name not in hub.tree:
            raise EntryNotFoundError(name)
        path = tmp_path / "download" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(hub.tree[name])
        return path

    monkeypatch.setattr(sync, "HfApi", lambda **kwargs: hub)
    monkeypatch.setattr(sync, "_download", download)
    return hub


def test_empty_seed_restores_without_manual_ingestion(shared_hub, tmp_path):
    seed_commit = sync.sync_news(NewsStore(tmp_path / "empty-seed"), seed=True)
    assert seed_commit != "initial"
    assert sync.sync_news(NewsStore(tmp_path / "empty-restore")) == seed_commit


@pytest.fixture
def seeded_store(shared_hub, tmp_path):
    store = NewsStore(tmp_path / "publisher")
    end = datetime(2026, 11, 23, 14, tzinfo=UTC)
    coverage = NewsCoverage(
        NEWS_PROVIDER,
        "AAA",
        end - timedelta(days=7),
        end,
        NEWS_SCHEMA_VERSION,
        NEWS_SENTIMENT_MODEL,
        NEWS_SENTIMENT_REVISION,
        "verified_empty",
        1,
        0,
        0,
        end,
        "proof",
    )
    store.commit_window(events=[], coverage=coverage, cache_rows=[])
    sync.sync_news(store, seed=True)
    return store, coverage


def test_publish_records_exact_local_checksums(seeded_store, shared_hub):
    store, _ = seeded_store
    manifest = json.loads(shared_hub.tree["news_v1/manifest.json"])
    with store._connect() as con:
        imports = dict(
            con.execute(
                "SELECT partition, checksum FROM news_sync_imports WHERE repo = ?",
                ["test/shared-news"],
            ).fetchall()
        )
    assert imports == {
        partition: entry["sha256"]
        for partition, entry in manifest["partitions"].items()
    }


@pytest.mark.parametrize("legacy_bookkeeping", [False, True])
def test_publisher_reuses_local_evidence_without_downloading(
    seeded_store, monkeypatch, legacy_bookkeeping
):
    store, _ = seeded_store
    if legacy_bookkeeping:
        store._with_write_lock(lambda con: con.execute("DELETE FROM news_sync_imports"))
    download = sync._download

    def manifest_only(repo, revision, name, token):
        assert name.endswith("manifest.json"), "publisher downloaded its own evidence"
        return download(repo, revision, name, token)

    monkeypatch.setattr(sync, "_download", manifest_only)
    sync.sync_news(store, publish=True)
    with store._connect() as con:
        assert con.execute("SELECT count(*) FROM news_sync_imports").fetchone() == (1,)


def test_restore_downloads_when_local_rows_differ(seeded_store, monkeypatch):
    store, coverage = seeded_store
    store.commit_window(
        events=[],
        coverage=replace(coverage, future_revision_excluded_count=1),
        cache_rows=[],
    )
    store._with_write_lock(lambda con: con.execute("DELETE FROM news_sync_imports"))
    download = sync._download
    requested = []

    def record_download(repo, revision, name, token):
        requested.append(name)
        return download(repo, revision, name, token)

    monkeypatch.setattr(sync, "_download", record_download)
    sync.sync_news(store)
    assert any(name.endswith(".parquet") for name in requested)
    window = NewsWindow(coverage.window_start_exclusive, coverage.window_end_inclusive)
    assert store.get_coverage("AAA", window) == coverage


def test_publish_preserves_writes_made_during_upload(
    seeded_store, shared_hub, monkeypatch
):
    store, coverage = seeded_store
    with store._connect() as con:
        previous_imports = con.execute("SELECT * FROM news_sync_imports").fetchall()
    store.commit_window(
        events=[], coverage=replace(coverage, symbol="BBB"), cache_rows=[]
    )
    create_commit = shared_hub.create_commit

    def write_during_commit(*args, **kwargs):
        result = create_commit(*args, **kwargs)
        store.commit_window(
            events=[], coverage=replace(coverage, symbol="CCC"), cache_rows=[]
        )
        return result

    monkeypatch.setattr(shared_hub, "create_commit", write_during_commit)
    sync.sync_news(store, publish=True)
    with store._connect() as con:
        assert (
            con.execute("SELECT * FROM news_sync_imports").fetchall()
            == previous_imports
        )
        assert con.execute("SELECT count(*) FROM news_sync_dirty").fetchone() == (1,)
    window = NewsWindow(coverage.window_start_exclusive, coverage.window_end_inclusive)
    assert store.get_coverage("CCC", window) is not None


def test_cold_store_skips_published_window(shared_hub, tmp_path):
    calls = []

    class Provider:
        def materialize(self, symbols, window):
            calls.append(list(symbols))
            return [], []

    end = datetime(2026, 2, 9, 14, tzinfo=UTC)
    start = end - timedelta(days=7)
    first = NewsStore(tmp_path / "host-one")
    digest = "a" * 64
    first.commit_window(
        events=[],
        coverage=NewsCoverage(
            provider=NEWS_PROVIDER,
            symbol="AAA",
            window_start_exclusive=start,
            window_end_inclusive=end,
            schema_version=NEWS_SCHEMA_VERSION,
            sentiment_model=NEWS_SENTIMENT_MODEL,
            sentiment_model_revision=NEWS_SENTIMENT_REVISION,
            status="verified_empty",
            page_count=1,
            event_count=0,
            future_revision_excluded_count=0,
            fetched_at=end,
            request_manifest_hash="proof",
        ),
        cache_rows=[
            (
                digest,
                NEWS_SENTIMENT_MODEL,
                NEWS_SENTIMENT_REVISION,
                1,
                0.5,
                0.6,
                0.1,
                0.3,
                0.6,
            )
        ],
    )
    sync.sync_news(first, seed=True)
    second = NewsStore(tmp_path / "host-two")
    run_backfill(
        symbols=["AAA"],
        start=start,
        end=end,
        store=second,
        service=Provider(),
        shared=True,
    )
    assert not calls
    assert second.get_coverage("AAA", NewsWindow(start, end)).status == "verified_empty"
    assert second.cache_get(digest) == (0.5, 0.6, 0.1, 0.3, 0.6)
    run_backfill(
        symbols=["AAA", "BBB"],
        start=start,
        end=end,
        store=second,
        service=Provider(),
        shared=True,
    )
    assert calls == [["BBB"]]


def test_session_news_uses_covered_intervals_and_keeps_unknown_missing(tmp_path):
    import pandas as pd

    from brain_api.news.patchtst import load_session_sentiment, session_news_window

    store = NewsStore(tmp_path)
    close = datetime(2026, 11, 27, 18, tzinfo=UTC)
    window = session_news_window(close)
    midpoint = close - timedelta(days=3)
    for start, end in ((window.start_exclusive, midpoint), (midpoint, close)):
        store.commit_window(
            events=[],
            coverage=NewsCoverage(
                provider=NEWS_PROVIDER,
                symbol="AAA",
                window_start_exclusive=start,
                window_end_inclusive=end,
                schema_version=NEWS_SCHEMA_VERSION,
                sentiment_model=NEWS_SENTIMENT_MODEL,
                sentiment_model_revision=NEWS_SENTIMENT_REVISION,
                status="verified_empty",
                page_count=1,
                event_count=0,
                future_revision_excluded_count=0,
                fetched_at=close,
                request_manifest_hash="test",
            ),
            cache_rows=[],
        )
    grid = pd.Series([close], index=pd.DatetimeIndex(["2026-11-27"]))
    sentiment = load_session_sentiment(store, ["AAA", "BBB"], grid)
    assert sentiment.loc["2026-11-27", "AAA"] == 0
    assert pd.isna(sentiment.loc["2026-11-27", "BBB"])


def test_coverage_merge_converges_and_preserves_local_rows(tmp_path):
    end = datetime(2026, 11, 23, 14, tzinfo=UTC)
    window = NewsWindow(end - timedelta(days=7), end)
    weak = NewsCoverage(
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
        1,
        end,
        "weak",
    )
    strong = replace(
        weak,
        status="complete",
        event_count=1,
        future_revision_excluded_count=0,
        request_manifest_hash="strong",
    )
    one, two = NewsStore(tmp_path / "one"), NewsStore(tmp_path / "two")
    one.commit_window(events=[], coverage=weak, cache_rows=[])
    one.commit_window(events=[], coverage=replace(weak, symbol="LOCAL"), cache_rows=[])
    two.commit_window(events=[], coverage=strong, cache_rows=[])
    partition = "news_coverage/2026-11"
    exported = []
    for index, store in enumerate((one, two)):
        path = tmp_path / f"{index}.parquet"
        with store._connect() as con:
            count = export_partition(con, partition, path)
        exported.append((path, count))
    for store, (path, count) in zip((one, two), reversed(exported), strict=True):
        store._with_write_lock(
            lambda con, path=path, count=count: import_partition(
                con, partition, path, count
            )
        )
        assert store.get_coverage("AAA", window) == strong
        assert store.get_coverage("LOCAL", window) is not None


def test_patchtst_job_reports_unusable_revisions_after_fetch(tmp_path):
    from brain_api.news.hashing import news_job_id
    from brain_api.news_etl.gaps import missing_windows
    from brain_api.news_etl.windows import requested_windows

    start = datetime(2026, 11, 9, 14, tzinfo=UTC)
    end = datetime(2026, 11, 23, 14, tzinfo=UTC)
    store = NewsStore(tmp_path)
    windows = requested_windows(start, end, "patchtst")
    assert windows

    class LateRevisionProvider:
        def materialize(self, symbols, window):
            for symbol in symbols:
                store.commit_window(
                    events=[],
                    coverage=NewsCoverage(
                        NEWS_PROVIDER,
                        symbol,
                        window.start_exclusive,
                        window.end_inclusive,
                        NEWS_SCHEMA_VERSION,
                        NEWS_SENTIMENT_MODEL,
                        NEWS_SENTIMENT_REVISION,
                        "verified_empty",
                        1,
                        0,
                        1,
                        end,
                        "late",
                    ),
                    cache_rows=[],
                )
            return [], []

    with pytest.raises(ValueError, match="remain unusable"):
        run_backfill(
            symbols=["AAA"],
            start=start,
            end=end,
            store=store,
            service=LateRevisionProvider(),
            purpose="patchtst",
        )
    job = store.get_job(
        news_job_id(start=start, end=end, symbols=["AAA"], purpose="patchtst")
    )
    assert job.status == "failed"
    assert missing_windows(
        store, symbols=["AAA"], start=start, end=end, purpose="patchtst"
    )

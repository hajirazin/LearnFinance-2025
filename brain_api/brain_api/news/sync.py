"""Synchronize only structured news evidence between local DuckDB and HF."""

from __future__ import annotations

import fcntl
import hashlib
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError

from brain_api.core.config import get_hf_news_sentiment_repo, get_hf_token
from brain_api.news.errors import NewsError
from brain_api.news.models import (
    NEWS_PROVIDER,
    NEWS_SCHEMA_VERSION,
    NEWS_SENTIMENT_MODEL,
    NEWS_SENTIMENT_REVISION,
    SCORING_SCHEMA_VERSION,
)
from brain_api.news.sync_partitions import (
    export_partition,
    import_partition,
    partition_query,
    seed_local_partitions,
)

if TYPE_CHECKING:
    from brain_api.news.store import NewsStore

_PREFIX = "news_v1"
_MANIFEST = f"{_PREFIX}/manifest.json"
_IDENTITY = {
    "schema": NEWS_SCHEMA_VERSION,
    "provider": NEWS_PROVIDER,
    "scorer": NEWS_SENTIMENT_MODEL,
    "scorer_revision": NEWS_SENTIMENT_REVISION,
    "scoring_schema": SCORING_SCHEMA_VERSION,
}


class NewsSyncError(NewsError):
    """The configured shared news evidence could not be synchronized."""


def _checksum(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _download(repo: str, revision: str, name: str, token: str | None) -> Path:
    return Path(
        hf_hub_download(repo, name, repo_type="dataset", revision=revision, token=token)
    )


def _read_manifest(repo: str, revision: str, token: str | None) -> dict | None:
    try:
        manifest = json.loads(_download(repo, revision, _MANIFEST, token).read_text())
    except EntryNotFoundError:
        return None
    if manifest.get("identity") != _IDENTITY or not isinstance(
        manifest.get("partitions"), dict
    ):
        raise ValueError("Unexpected news dataset manifest identity")
    partitions = manifest["partitions"]
    for partition, entry in partitions.items():
        partition_query(partition)
        if (
            not isinstance(entry, dict)
            or not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256")))
            or type(entry.get("rows")) is not int
            or entry["rows"] < 0
        ):
            raise ValueError(f"Invalid manifest entry for {partition}")
    return partitions


def _restore(
    store: NewsStore, repo: str, revision: str, partitions: dict, token: str | None
) -> None:
    with store._connect() as con:
        imported = dict(
            con.execute(
                "SELECT partition, checksum FROM news_sync_imports WHERE repo = ?",
                [repo],
            ).fetchall()
        )
    pending = {}
    for partition, entry in partitions.items():
        if imported.get(partition) == entry["sha256"]:
            continue
        path = _download(repo, revision, f"{_PREFIX}/{partition}.parquet", token)
        if _checksum(path) != entry["sha256"]:
            raise ValueError(f"News partition checksum mismatch for {partition}")
        pending[partition] = path
    if not pending:
        return

    def merge(con):
        for partition, path in pending.items():
            entry = partitions[partition]
            import_partition(con, partition, path, entry["rows"])
            con.execute(
                "INSERT OR REPLACE INTO news_sync_imports VALUES (?, ?, ?)",
                [repo, partition, entry["sha256"]],
            )

    store._with_write_lock(merge)


def _publish(
    store: NewsStore, api: HfApi, repo: str, revision: str, partitions: dict
) -> str:
    with TemporaryDirectory(prefix="news-sync-") as temporary:
        root = Path(temporary)
        # One read transaction captures the export and its dirty generations.
        with store._connect() as con:
            con.execute("BEGIN")
            dirty = con.execute(
                "SELECT partition, generation FROM news_sync_dirty ORDER BY partition"
            ).fetchall()
            operations = []
            manifest = dict(partitions)
            for index, (partition, _) in enumerate(dirty):
                path = root / f"{index}.parquet"
                rows = export_partition(con, partition, path)
                entry = {"sha256": _checksum(path), "rows": rows}
                if entry != manifest.get(partition):
                    operations.append(
                        CommitOperationAdd(f"{_PREFIX}/{partition}.parquet", path)
                    )
                    manifest[partition] = entry
            con.execute("COMMIT")
        if operations or not partitions:
            operations.append(
                CommitOperationAdd(
                    _MANIFEST,
                    json.dumps(
                        {"identity": _IDENTITY, "partitions": manifest}, sort_keys=True
                    ).encode(),
                )
            )
            revision = api.create_commit(
                repo,
                operations=operations,
                repo_type="dataset",
                parent_commit=revision,
                commit_message="Sync scored news and coverage",
            ).oid

        def mark_published(con):
            for partition, generation in dirty:
                con.execute(
                    "DELETE FROM news_sync_dirty WHERE partition = ? AND generation = ?",
                    [partition, generation],
                )

        store._with_write_lock(mark_published)
    return revision


def sync_news(
    store: NewsStore | None = None, *, publish: bool = False, seed: bool = False
) -> str | None:
    """Restore before historical fetch; explicitly seed an unpublished dataset."""
    repo = get_hf_news_sentiment_repo()
    if not repo:
        if seed:
            raise NewsSyncError("Set HF_NEWS_SENTIMENT_REPO before seeding news")
        return None
    if store is None:
        from brain_api.news.store import NewsStore

        store = NewsStore()
    token = get_hf_token() or None
    api = HfApi(token=token)
    sync_lock = store.db_path.with_suffix(".sync.lock")
    try:
        with sync_lock.open("a") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            if seed:
                api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
            for attempt in range(3):
                info = api.repo_info(repo, repo_type="dataset")
                if info.private is not True or not info.sha:
                    raise ValueError(
                        "Shared news requires a private dataset with a readable head"
                    )
                partitions = _read_manifest(repo, info.sha, token)
                if partitions is None and not seed:
                    raise ValueError(
                        "Shared news dataset has no published partitions; seed it first"
                    )
                partitions = partitions or {}
                store._with_write_lock(lambda con: seed_local_partitions(con, repo))
                _restore(store, repo, info.sha, partitions, token)
                if not publish and not seed:
                    return info.sha
                try:
                    return _publish(store, api, repo, info.sha, partitions)
                except HfHubHTTPError as exc:
                    if (
                        attempt == 2
                        or exc.response is None
                        or exc.response.status_code not in (409, 412)
                    ):
                        raise
    except ValueError as exc:
        raise NewsSyncError(str(exc)) from exc
    except HfHubHTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        raise NewsSyncError(
            f"Hugging Face news sync failed with status {status}"
        ) from exc
    except Exception as exc:
        raise NewsSyncError(
            f"Could not synchronize shared news: {type(exc).__name__}"
        ) from exc
    raise NewsSyncError("Could not publish shared news after concurrent changes")

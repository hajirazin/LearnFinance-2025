"""Synchronize only structured news evidence between local DuckDB and HF."""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import re
import time
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
logger = logging.getLogger(__name__)
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


def _recognize_local_partitions(
    store: NewsStore, repo: str, partitions: dict
) -> set[str]:
    """Record exact local matches before requesting remote evidence."""
    if not partitions:
        return set()
    with TemporaryDirectory(prefix="news-verify-") as temporary:
        root = Path(temporary)

        def recognize(con):
            matched = set()
            for index, (partition, entry) in enumerate(partitions.items(), start=1):
                path = root / f"{index}.parquet"
                rows = export_partition(con, partition, path)
                if rows == entry["rows"] and _checksum(path) == entry["sha256"]:
                    con.execute(
                        "INSERT OR REPLACE INTO news_sync_imports VALUES (?, ?, ?)",
                        [repo, partition, entry["sha256"]],
                    )
                    matched.add(partition)
                if index % 25 == 0 or index == len(partitions):
                    logger.info(
                        "news sync checking local evidence partitions=%s/%s matched=%s",
                        index,
                        len(partitions),
                        len(matched),
                    )
            return matched

        return store._with_write_lock(recognize)


def _restore(
    store: NewsStore, repo: str, revision: str, partitions: dict, token: str | None
) -> None:
    started = time.monotonic()
    with store._connect() as con:
        imported = dict(
            con.execute(
                "SELECT partition, checksum FROM news_sync_imports WHERE repo = ?",
                [repo],
            ).fetchall()
        )
    candidates = {
        key: entry
        for key, entry in partitions.items()
        if imported.get(key) != entry["sha256"]
    }
    matched = _recognize_local_partitions(store, repo, candidates)
    imported.update({key: candidates[key]["sha256"] for key in matched})
    pending = {}
    total = len(candidates) - len(matched)
    logger.info(
        "news sync restore start partitions=%s unchanged=%s",
        total,
        len(partitions) - total,
    )
    for partition, entry in partitions.items():
        if imported.get(partition) == entry["sha256"]:
            continue
        path = _download(repo, revision, f"{_PREFIX}/{partition}.parquet", token)
        if _checksum(path) != entry["sha256"]:
            raise ValueError(f"News partition checksum mismatch for {partition}")
        pending[partition] = path
        if len(pending) % 25 == 0 or len(pending) == total:
            logger.info(
                "news sync download progress partitions=%s/%s latest=%s elapsed_s=%.1f",
                len(pending),
                total,
                partition,
                time.monotonic() - started,
            )
    if not pending:
        logger.info("news sync restore up to date")
        return

    def merge(con):
        logger.info("news sync merge start partitions=%s", len(pending))
        for index, (partition, path) in enumerate(pending.items(), start=1):
            entry = partitions[partition]
            import_partition(con, partition, path, entry["rows"])
            con.execute(
                "INSERT OR REPLACE INTO news_sync_imports VALUES (?, ?, ?)",
                [repo, partition, entry["sha256"]],
            )
            if index % 25 == 0 or index == len(pending):
                logger.info(
                    "news sync merge progress partitions=%s/%s latest=%s elapsed_s=%.1f",
                    index,
                    len(pending),
                    partition,
                    time.monotonic() - started,
                )

    store._with_write_lock(merge)
    logger.info(
        "news sync restore completed partitions=%s elapsed_s=%.1f",
        len(pending),
        time.monotonic() - started,
    )


def _publish(
    store: NewsStore, api: HfApi, repo: str, revision: str, partitions: dict
) -> str:
    started = time.monotonic()
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
            logger.info("news sync export start partitions=%s", len(dirty))
            for index, (partition, _) in enumerate(dirty):
                path = root / f"{index}.parquet"
                rows = export_partition(con, partition, path)
                entry = {"sha256": _checksum(path), "rows": rows}
                if entry != manifest.get(partition):
                    operations.append(
                        CommitOperationAdd(f"{_PREFIX}/{partition}.parquet", path)
                    )
                    manifest[partition] = entry
                if (index + 1) % 25 == 0 or index + 1 == len(dirty):
                    logger.info(
                        "news sync export progress partitions=%s/%s latest=%s elapsed_s=%.1f",
                        index + 1,
                        len(dirty),
                        partition,
                        time.monotonic() - started,
                    )
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
            logger.info(
                "news sync upload start files=%s repo=%s", len(operations), repo
            )
            revision = api.create_commit(
                repo,
                operations=operations,
                repo_type="dataset",
                parent_commit=revision,
                commit_message="Sync scored news and coverage",
            ).oid
            logger.info(
                "news sync upload completed commit=%s elapsed_s=%.1f",
                revision,
                time.monotonic() - started,
            )

        def mark_published(con):
            for partition, generation in dirty:
                current = con.execute(
                    "SELECT generation FROM news_sync_dirty WHERE partition = ?",
                    [partition],
                ).fetchone()
                if current != (generation,):
                    continue
                con.execute(
                    "INSERT OR REPLACE INTO news_sync_imports VALUES (?, ?, ?)",
                    [repo, partition, manifest[partition]["sha256"]],
                )
                con.execute(
                    "DELETE FROM news_sync_dirty WHERE partition = ? AND generation = ?",
                    [partition, generation],
                )

        store._with_write_lock(mark_published)
    logger.info(
        "news sync publish completed commit=%s elapsed_s=%.1f",
        revision,
        time.monotonic() - started,
    )
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
    started = time.monotonic()
    action = "seed" if seed else "publish" if publish else "restore"
    logger.info("news sync start action=%s repo=%s", action, repo)
    sync_lock = store.db_path.with_suffix(".sync.lock")
    try:
        with sync_lock.open("a") as handle:
            logger.info("news sync waiting for local sync lock")
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            logger.info("news sync acquired local sync lock")
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
                logger.info(
                    "news sync manifest loaded commit=%s partitions=%s",
                    info.sha,
                    len(partitions),
                )
                logger.info("news sync indexing local evidence")
                store._with_write_lock(lambda con: seed_local_partitions(con, repo))
                logger.info("news sync local evidence indexed")
                _restore(store, repo, info.sha, partitions, token)
                if not publish and not seed:
                    logger.info(
                        "news sync completed action=%s commit=%s elapsed_s=%.1f",
                        action,
                        info.sha,
                        time.monotonic() - started,
                    )
                    return info.sha
                try:
                    revision = _publish(store, api, repo, info.sha, partitions)
                    logger.info(
                        "news sync completed action=%s commit=%s elapsed_s=%.1f",
                        action,
                        revision,
                        time.monotonic() - started,
                    )
                    return revision
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

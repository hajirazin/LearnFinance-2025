"""Delete every hashed forecaster snapshot on local disk, Hugging Face, or both."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brain_api.storage.forecaster_snapshots.local import SnapshotLocalStorage


class SnapshotStorageTarget(StrEnum):
    """Backend selected by the required ``storage`` query argument."""

    LOCAL = "local"
    HF = "hf"
    BOTH = "both"


class HuggingFaceSnapshotRepoNotConfiguredError(Exception):
    """Raised when the target includes Hugging Face and the bucket has no repo."""


@dataclass(frozen=True)
class DeletedForecasterSnapshot:
    """One hashed snapshot this call actually removed on at least one backend."""

    cutoff_date: date
    snapshot_digest: str
    deleted_local: bool
    deleted_hf: bool


@dataclass(frozen=True)
class DeleteForecasterSnapshotsResult:
    """Successful delete of every hashed snapshot for one bucket and target."""

    forecaster_bucket: str
    storage: SnapshotStorageTarget
    deleted: tuple[DeletedForecasterSnapshot, ...]


@dataclass(frozen=True)
class ForecasterSnapshotDeleteFailure:
    """One backend delete that raised after other snapshots may already be gone."""

    cutoff_date: date
    snapshot_digest: str
    backend: str
    error: str


class ForecasterSnapshotDeleteError(Exception):
    """At least one hashed snapshot could not be deleted.

    ``deleted`` lists identities removed on at least one backend before or
    beside the failures. Already-absent paths are omitted from both lists.
    """

    def __init__(
        self,
        *,
        deleted: tuple[DeletedForecasterSnapshot, ...],
        failures: tuple[ForecasterSnapshotDeleteFailure, ...],
    ) -> None:
        self.deleted = deleted
        self.failures = failures
        super().__init__(f"Failed to delete {len(failures)} forecaster snapshot(s)")


def delete_all_forecaster_snapshots(
    storage: SnapshotLocalStorage,
    target: SnapshotStorageTarget,
) -> DeleteForecasterSnapshotsResult:
    """Delete every hashed snapshot on ``target``.

    ``local`` touches disk only. ``hf`` touches Hugging Face branches only.
    ``both`` requires a configured HF repo before any delete, then removes
    each identity from the union of the two inventories. An identity present
    on only one side is removed there; the other side already being absent
    is success.

    Raises:
        HuggingFaceSnapshotRepoNotConfiguredError: ``hf`` or ``both`` and
            the bucket has no HF repo. Nothing is deleted.
        Exception: Listing HF branches fails. Nothing is deleted.
        ForecasterSnapshotDeleteError: A per-snapshot delete failed. Earlier
            successful deletes in this call stay deleted.
    """
    includes_local = target in (SnapshotStorageTarget.LOCAL, SnapshotStorageTarget.BOTH)
    includes_hf = target in (SnapshotStorageTarget.HF, SnapshotStorageTarget.BOTH)

    hf_identities = storage.list_hf_snapshot_identities_strict() if includes_hf else []
    local_identities = (
        storage.list_local_snapshot_identities() if includes_local else []
    )

    identities = sorted(set(local_identities) | set(hf_identities))
    deleted: list[DeletedForecasterSnapshot] = []
    failures: list[ForecasterSnapshotDeleteFailure] = []

    for cutoff, digest in identities:
        deleted_local = False
        deleted_hf = False
        if includes_local:
            try:
                deleted_local = storage.delete_local_snapshot(cutoff, digest)
            except Exception as exc:
                failures.append(
                    ForecasterSnapshotDeleteFailure(
                        cutoff_date=cutoff,
                        snapshot_digest=digest,
                        backend="local",
                        error=str(exc),
                    )
                )
        if includes_hf:
            try:
                deleted_hf = storage.delete_hf_snapshot(cutoff, digest)
            except Exception as exc:
                failures.append(
                    ForecasterSnapshotDeleteFailure(
                        cutoff_date=cutoff,
                        snapshot_digest=digest,
                        backend="hf",
                        error=str(exc),
                    )
                )
        if deleted_local or deleted_hf:
            deleted.append(
                DeletedForecasterSnapshot(
                    cutoff_date=cutoff,
                    snapshot_digest=digest,
                    deleted_local=deleted_local,
                    deleted_hf=deleted_hf,
                )
            )

    deleted_tuple = tuple(deleted)
    if failures:
        raise ForecasterSnapshotDeleteError(
            deleted=deleted_tuple,
            failures=tuple(failures),
        )
    return DeleteForecasterSnapshotsResult(
        forecaster_bucket=storage.forecaster_type,
        storage=target,
        deleted=deleted_tuple,
    )

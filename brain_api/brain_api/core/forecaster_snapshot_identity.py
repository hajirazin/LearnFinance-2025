"""Stable snapshot identity for SAC walk-forward and forecaster training.

Snapshot folder and branch identity depends only on the canonical forecaster
bucket, cutoff, and config. The training symbol slate and price-loading window
remain training inputs, but they do not affect snapshot lookup identity.

This module is also the read-side mirror of the snapshot backfill loops:
:func:`annual_snapshot_cutoffs` and :func:`annual_snapshot_parity` decide
which December 31 digests are ready, need a copy, or need training.
:func:`count_missing_snapshots` classifies that parity and does not upload
or download. :func:`_resolve_check_hf` still translates
:class:`~brain_api.storage.policy.StoragePolicy` for SAC
``ensure_snapshot_available``; the annual loop does not call it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any, Literal

from brain_api.core.lstm.config import DEFAULT_CONFIG as LSTM_DEFAULT_CONFIG
from brain_api.core.patchtst.config import DEFAULT_CONFIG as PATCHTST_DEFAULT_CONFIG
from brain_api.core.version import compute_snapshot_identity_hash
from brain_api.storage.policy import (
    StoragePolicy,
    StoragePolicyError,
)

if TYPE_CHECKING:
    from brain_api.storage.forecaster_snapshots.local import SnapshotLocalStorage


def expected_dec31_walkforward_snapshot_hash(
    *,
    forecaster_bucket: str,
    cutoff_date: date,
    config_dict: dict[str, Any],
) -> str:
    """12-char digest for ``snapshot-{cutoff}-{digest}/`` (Dec-31 backfill rows)."""
    return compute_snapshot_identity_hash(forecaster_bucket, cutoff_date, config_dict)


def lstm_walkforward_expectation_bundle() -> tuple[str, dict[str, Any]]:
    """Return the identity inputs for standalone LSTM walk-forward snapshots."""
    return "lstm_halal_new", LSTM_DEFAULT_CONFIG.to_dict()


def patchtst_walkforward_expectation_bundle() -> tuple[str, dict[str, Any]]:
    """Return the identity inputs for PatchTST walk-forward snapshots."""
    return "patchtst_halal_new", PATCHTST_DEFAULT_CONFIG.to_dict()


SnapshotParity = Literal["ready", "upload", "download", "train"]


def annual_snapshot_cutoffs(start_date: date, end_date: date) -> list[date]:
    """December 31 cutoffs SAC can consume inside ``[start_date, end_date]``.

    The first cutoff is December 31 of ``start_date.year - 1``. A cutoff is
    included only when that December 31 is on or before ``end_date``. The
    training window's raw end date is not a cutoff unless it is that
    December 31.
    """
    first_snapshot_year = start_date.year - 1
    cutoffs: list[date] = []
    for year in range(first_snapshot_year, end_date.year + 1):
        cutoff = date(year, 12, 31)
        if cutoff <= end_date:
            cutoffs.append(cutoff)
    return cutoffs


def annual_snapshot_parity(
    storage: SnapshotLocalStorage,
    cutoff_date: date,
    snapshot_digest: str,
) -> SnapshotParity:
    """How to obtain one expected annual digest without reading storage policy.

    A rejected audit copy and a different digest for the same date do not
    count. No Hugging Face repo means a canonical local directory is enough.
    """
    local = storage.snapshot_exists(cutoff_date, snapshot_digest)
    repo = storage._get_hf_repo()
    on_hf = (
        storage.snapshot_digest_exists_on_hf(cutoff_date, snapshot_digest)
        if repo
        else False
    )
    if local and (not repo or on_hf):
        return "ready"
    if local and repo and not on_hf:
        return "upload"
    if not local and repo and on_hf:
        return "download"
    return "train"


# ---------------------------------------------------------------------------
# Policy translator + missing-snapshot inventory
# ---------------------------------------------------------------------------


def _resolve_check_hf(
    *,
    snapshot_storage: SnapshotLocalStorage,
    policy: StoragePolicy,
) -> bool:
    """Translate ``StoragePolicy`` + HF repo presence into the
    ``check_hf`` flag accepted by ``snapshot_exists_anywhere``.

    Used by SAC snapshot availability. Annual training inventory and
    backfill call :func:`annual_snapshot_parity` instead, so this
    translator does not decide whether a December 31 digest is trained.

    * ``hf_first`` + no HF repo configured for this bucket -> raises
      :class:`StoragePolicyError`. Per AGENTS.md rule #1 (no silent
      fallback): the operator selected ``hf_first`` and there is no
      HF endpoint to consult, so the request must fail loudly rather
      than degrade to local-only.
    * ``hf_first`` + HF repo configured -> ``True``.
    * ``local_first`` + HF repo configured -> ``True`` (HF is the
      fallback for a wiped local cache; matches the long-standing
      behaviour of the backfill loops).
    * ``local_first`` + no HF repo -> ``False`` (local-only mode).
    """
    hf_repo = snapshot_storage._get_hf_repo()
    if policy is StoragePolicy.HF_FIRST and not hf_repo:
        raise StoragePolicyError(
            f"hf_first policy requires HF repo for snapshot bucket "
            f"{snapshot_storage.forecaster_type!r}; got none. Set the "
            f"bucket's HF env var or switch STORAGE_BACKEND to local_first."
        )
    return hf_repo is not None


@dataclass(frozen=True)
class MissingSnapshotInventory:
    """Annual cutoffs whose parity is upload, download, or train.

    ``historical_cutoffs`` is the ordered tuple of December 31 dates that
    are not yet ``ready`` on both sides (or locally, when no HF repo is
    configured). The training window end date is not a cutoff.
    """

    historical_cutoffs: tuple[date, ...]

    @property
    def is_empty(self) -> bool:
        return not self.historical_cutoffs

    @property
    def total_missing(self) -> int:
        return len(self.historical_cutoffs)


def count_missing_snapshots(
    *,
    forecaster_type: str,
    train_window: tuple[date, date],
    config_dict: dict[str, Any],
    snapshot_storage: SnapshotLocalStorage,
    policy: StoragePolicy | None = None,
) -> MissingSnapshotInventory:
    """Read-side mirror of ``_backfill_lstm_snapshots`` /
    ``_backfill_patchtst_snapshots`` -- returns *which* snapshots are
    missing without training anything.

    Used by the training routes' synchronous "any backfill needed?"
    scan that decides between returning a 200 cached response and
    enqueuing a snapshots-only background job. The scan classifies
    parity and does not upload or download.

    Math correctness invariant: every cutoff uses
    :func:`compute_snapshot_identity_hash`, bit-identical to the writer.
    The training window determines which December 31 cutoffs to inventory,
    but its raw end date and the training symbols are not snapshot
    identity inputs.

    Args:
        forecaster_type: Canonical snapshot bucket name.
        train_window: ``(start_date, end_date)`` from
            :func:`brain_api.core.config.resolve_training_window`.
        config_dict: Forecaster config as a plain dict.
        snapshot_storage: Bucket storage instance used to probe local
            and Hugging Face presence.
        policy: Ignored. Annual parity does not consult ``STORAGE_BACKEND``.

    Returns:
        :class:`MissingSnapshotInventory` of December 31 cutoffs that are
        not yet ``ready``.
    """
    del policy
    start_date, end_date = train_window
    historical: list[date] = []
    for cutoff_date in annual_snapshot_cutoffs(start_date, end_date):
        digest = compute_snapshot_identity_hash(
            forecaster_type, cutoff_date, config_dict
        )
        if annual_snapshot_parity(snapshot_storage, cutoff_date, digest) != "ready":
            historical.append(cutoff_date)

    return MissingSnapshotInventory(historical_cutoffs=tuple(historical))

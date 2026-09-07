"""Stable snapshot identity for SAC walk-forward and forecaster training.

Snapshot folder and branch identity depends only on the canonical forecaster
bucket, cutoff, and config. The training symbol slate and price-loading window
remain training inputs, but they do not affect snapshot lookup identity.

This module is also the read-side mirror of the snapshot backfill loops:
:func:`count_missing_snapshots` answers "which snapshots would the backfill
need to train?" without touching the trainer code, and
:func:`_resolve_check_hf` is the single source of truth that translates
:class:`~brain_api.storage.policy.StoragePolicy` + HF repo presence into
the boolean accepted by
:meth:`~brain_api.storage.forecaster_snapshots.local.SnapshotLocalStorage.snapshot_exists_anywhere`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any

from brain_api.core.lstm.config import DEFAULT_CONFIG as LSTM_DEFAULT_CONFIG
from brain_api.core.patchtst.config import DEFAULT_CONFIG as PATCHTST_DEFAULT_CONFIG
from brain_api.core.version import compute_snapshot_identity_hash
from brain_api.storage.policy import (
    StoragePolicy,
    StoragePolicyError,
    get_storage_policy,
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

    Single source of truth. Mirrors
    :func:`brain_api.storage.forecaster_snapshots.local.SnapshotLocalStorage.ensure_snapshot_available`
    so every existence-check call site behaves identically:

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
    """Snapshots that exist neither locally nor (per the storage policy)
    on HuggingFace.

    ``end_window_cutoff`` is ``None`` when the end-of-window snapshot
    (the one piggybacked on main training) is present. ``historical_cutoffs``
    is the ordered tuple of Dec-31 backfill cutoffs that are missing.
    """

    end_window_cutoff: date | None
    historical_cutoffs: tuple[date, ...]

    @property
    def is_empty(self) -> bool:
        return self.end_window_cutoff is None and not self.historical_cutoffs

    @property
    def total_missing(self) -> int:
        return (1 if self.end_window_cutoff is not None else 0) + len(
            self.historical_cutoffs
        )


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
    enqueuing a snapshots-only background job.

    Math correctness invariant: every cutoff uses
    :func:`compute_snapshot_identity_hash`, bit-identical to the writer.
    The training window determines which cutoffs to inventory, but its start
    and the training symbols are not snapshot identity inputs.

    Args:
        forecaster_type: Canonical snapshot bucket name.
        train_window: ``(start_date, end_date)`` from
            :func:`brain_api.core.config.resolve_training_window`.
        config_dict: Forecaster config as a plain dict.
        snapshot_storage: Bucket storage instance used to probe local
            and (per policy) HF presence.
        policy: Optional override; when ``None`` resolves via
            :func:`get_storage_policy` (i.e. ``STORAGE_BACKEND`` env).

    Returns:
        :class:`MissingSnapshotInventory` describing which cutoffs
        need to be created to fully populate the bucket.

    Raises:
        StoragePolicyError: when ``hf_first`` is active and the
            bucket has no HF repo configured. Surfaced from
            :func:`_resolve_check_hf` so callers can map it to a 503.
    """
    if policy is None:
        policy = get_storage_policy()
    check_hf = _resolve_check_hf(snapshot_storage=snapshot_storage, policy=policy)

    start_date, end_date = train_window

    end_window_digest = compute_snapshot_identity_hash(
        forecaster_type, end_date, config_dict
    )
    end_window_present = snapshot_storage.snapshot_exists_anywhere(
        end_date,
        end_window_digest,
        check_hf=check_hf,
    )
    end_window_cutoff: date | None = None if end_window_present else end_date

    start_year = start_date.year
    end_year = end_date.year
    first_snapshot_year = start_year - 1
    historical: list[date] = []
    for year in range(first_snapshot_year, end_year):
        cutoff_date = date(year, 12, 31)
        backfill_digest = compute_snapshot_identity_hash(
            forecaster_type, cutoff_date, config_dict
        )
        if not snapshot_storage.snapshot_exists_anywhere(
            cutoff_date,
            backfill_digest,
            check_hf=check_hf,
        ):
            historical.append(cutoff_date)

    return MissingSnapshotInventory(
        end_window_cutoff=end_window_cutoff,
        historical_cutoffs=tuple(historical),
    )

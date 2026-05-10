"""Stable snapshot-hash inputs for SAC US walk-forward and forecaster trains.

Monthly forecaster snapshots for year-end cutoffs use the extended backfill
price window start (matching ``_backfill_lstm_snapshots`` /
``_backfill_patchtst_snapshots``). SAC dual-forecast generation only consumes
those Dec-31 checkpoints from the ``lstm_halal_new`` and ``patchtst_halal_new``
buckets via :func:`~brain_api.storage.policy.ensure_snapshot_for_bucket`.

This module is also the read-side mirror of the snapshot backfill loops:
:func:`count_missing_snapshots` answers "which snapshots would the backfill
need to train?" without touching the trainer code, and
:func:`_resolve_check_hf` is the single source of truth that translates
:class:`~brain_api.storage.policy.StoragePolicy` + HF repo presence into
the boolean accepted by
:meth:`~brain_api.storage.forecaster_snapshots.local.SnapshotLocalStorage.snapshot_exists_anywhere`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

from brain_api.core.config import resolve_training_window
from brain_api.core.lstm.config import DEFAULT_CONFIG as LSTM_DEFAULT_CONFIG
from brain_api.core.model_buckets import ModelType, get_bucket
from brain_api.core.patchtst.config import DEFAULT_CONFIG as PATCHTST_DEFAULT_CONFIG
from brain_api.core.version import compute_model_hash
from brain_api.storage.policy import (
    StoragePolicy,
    StoragePolicyError,
    get_storage_policy,
)

if TYPE_CHECKING:
    from brain_api.storage.forecaster_snapshots.local import SnapshotLocalStorage

# Must stay aligned with ``bootstrap_years`` in the LSTM/PatchTST backfill loops.
_SNAPSHOT_BACKFILL_BOOTSTRAP_YEARS = 4


def extended_backfill_window_start_date() -> date:
    """First calendar day loaded for RL snapshot backfill (extended window).

    Mirrors ``routes/training/lstm.py::_backfill_lstm_snapshots`` and
    ``routes/training/patchtst.py::_backfill_patchtst_snapshots``.
    """
    start_date, _ = resolve_training_window()
    start_year = start_date.year
    first_snapshot_year = start_year - 1
    return date(
        first_snapshot_year - _SNAPSHOT_BACKFILL_BOOTSTRAP_YEARS,
        1,
        1,
    )


def halal_new_lstm_resolver_symbols() -> list[str]:
    return list(get_bucket(ModelType.LSTM, "halal_new").symbols_resolver())


def halal_new_patchtst_resolver_symbols() -> list[str]:
    return list(get_bucket(ModelType.PATCHTST, "halal_new").symbols_resolver())


def expected_dec31_walkforward_snapshot_hash(
    *,
    forecaster_bucket: str,
    cutoff_date: date,
    resolver_symbols: list[str],
    config_dict: dict,
) -> str:
    """12-char digest for ``snapshot-{cutoff}-{digest}/`` (Dec-31 backfill rows)."""
    window_start = extended_backfill_window_start_date()
    return compute_model_hash(
        forecaster_bucket,
        window_start,
        cutoff_date,
        resolver_symbols,
        config_dict,
    )


def lstm_walkforward_expectation_bundle() -> tuple[str, list[str], dict]:
    """(bucket_name, resolver_symbols, default_lstm_config_dict) for SAC LSTM snaps."""
    return (
        "lstm_halal_new",
        halal_new_lstm_resolver_symbols(),
        LSTM_DEFAULT_CONFIG.to_dict(),
    )


def patchtst_walkforward_expectation_bundle() -> tuple[str, list[str], dict]:
    """(bucket_name, resolver_symbols, default_patchtst_config_dict)."""
    return (
        "patchtst_halal_new",
        halal_new_patchtst_resolver_symbols(),
        PATCHTST_DEFAULT_CONFIG.to_dict(),
    )


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
    symbols: Sequence[str],
    config_dict: dict,
    snapshot_storage: SnapshotLocalStorage,
    policy: StoragePolicy | None = None,
) -> MissingSnapshotInventory:
    """Read-side mirror of ``_backfill_lstm_snapshots`` /
    ``_backfill_patchtst_snapshots`` -- returns *which* snapshots are
    missing without training anything.

    Used by the training routes' synchronous "any backfill needed?"
    scan that decides between returning a 200 cached response and
    enqueuing a snapshots-only background job.

    Math correctness invariant (AGENTS.md rule #2): the digest formulas
    here MUST stay bit-identical to the ones in the backfill loops.
    Two formulas are used because the existing main-training pipeline
    already uses two:

    * End-of-window snapshot uses ``compute_model_hash(forecaster_type,
      start_date, end_date, symbols, config_dict)`` -- the resolved
      training window from :func:`resolve_training_window`.
    * Historical backfill snapshots (Dec-31 of each ``year`` in
      ``range(start_year - 1, end_year)``) use
      ``compute_model_hash(forecaster_type, snapshot_data_start,
      cutoff_date, symbols, config_dict)`` where ``snapshot_data_start
      = date(start_year - 1 - bootstrap_years, 1, 1)``.

    If the backfill loops ever change a digest input, this function
    must change in lockstep.

    Args:
        forecaster_type: Snapshot bucket name (e.g. ``"lstm_halal_new"``);
            also used as the ``forecaster_type`` argument to
            ``compute_model_hash``.
        train_window: ``(start_date, end_date)`` from
            :func:`brain_api.core.config.resolve_training_window`.
        symbols: Stock symbols passed to the trainer (in the order
            ``compute_model_hash`` will see them).
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
    symbols_list = list(symbols)

    # End-of-window snapshot (digest matches the existing main-training
    # pipeline's end-window write block, NOT the backfill formula).
    end_window_digest = compute_model_hash(
        forecaster_type,
        start_date,
        end_date,
        symbols_list,
        config_dict,
    )
    end_window_present = snapshot_storage.snapshot_exists_anywhere(
        end_date,
        end_window_digest,
        check_hf=check_hf,
    )
    end_window_cutoff: date | None = None if end_window_present else end_date

    # Historical Dec-31 backfill snapshots (digest matches the existing
    # backfill loops' formula, with the extended window start).
    start_year = start_date.year
    end_year = end_date.year
    first_snapshot_year = start_year - 1
    snapshot_data_start = date(
        first_snapshot_year - _SNAPSHOT_BACKFILL_BOOTSTRAP_YEARS,
        1,
        1,
    )
    historical: list[date] = []
    for year in range(first_snapshot_year, end_year):
        cutoff_date = date(year, 12, 31)
        backfill_digest = compute_model_hash(
            forecaster_type,
            snapshot_data_start,
            cutoff_date,
            symbols_list,
            config_dict,
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

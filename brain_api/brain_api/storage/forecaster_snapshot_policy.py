"""Storage-policy adapter for canonical forecaster walk-forward snapshots."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brain_api.storage.forecaster_snapshots.local import SnapshotLocalStorage
    from brain_api.storage.policy import StoragePolicy


def ensure_snapshot_for_bucket(
    *,
    snapshot_storage: SnapshotLocalStorage,
    cutoff_date: date,
    policy: StoragePolicy | None = None,
) -> bool:
    """Ensure the expected hashed forecaster snapshot is available locally.

    Walk-forward identity uses the canonical snapshot bucket, cutoff, and
    default forecaster config. Folder and HF branch names are
    ``snapshot-{cutoff}-{digest}``.
    """
    from brain_api.core.forecaster_snapshot_identity import (
        expected_dec31_walkforward_snapshot_hash,
        lstm_walkforward_expectation_bundle,
        patchtst_walkforward_expectation_bundle,
    )
    from brain_api.storage.policy import StoragePolicyError

    bucket_type = snapshot_storage.forecaster_type
    if bucket_type == "lstm_halal_new":
        identity_bucket, wf_cfg = lstm_walkforward_expectation_bundle()
    elif bucket_type == "patchtst_halal_new":
        identity_bucket, wf_cfg = patchtst_walkforward_expectation_bundle()
    else:
        raise StoragePolicyError(
            "Walk-forward snapshot ensure is wired only for lstm_halal_new and "
            f"patchtst_halal_new; got {bucket_type!r}"
        )

    snapshot_digest = expected_dec31_walkforward_snapshot_hash(
        forecaster_bucket=identity_bucket,
        cutoff_date=cutoff_date,
        config_dict=wf_cfg,
    )

    return snapshot_storage.ensure_snapshot_available(
        cutoff_date, snapshot_digest, policy=policy
    )

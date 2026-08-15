"""Canonical-vs-rejected persist for forecaster snapshots.

Metrics are checked before any sibling eviction. Healthy models go to
``snapshot-{cutoff}-{digest}/`` (and HF). Unhealthy models go to
``rejected/snapshot-{cutoff}-{digest}/`` so artifacts survive the run
without replacing a valid canonical snapshot.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from sklearn.preprocessing import StandardScaler

from brain_api.core.training_utils import evaluate_forecaster_artifact_health
from brain_api.storage.forecaster_snapshots import SnapshotLocalStorage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SnapshotPersistResult:
    """Outcome of :func:`persist_forecaster_snapshot`."""

    is_canonical: bool
    failure_reasons: list[str]
    path: Path


def persist_forecaster_snapshot(
    *,
    snapshot_storage: SnapshotLocalStorage,
    cutoff_date: date,
    snapshot_digest: str,
    model: Any,
    feature_scaler: StandardScaler,
    config: Any,
    metadata: dict[str, Any],
    train_loss: float,
    val_loss: float,
    snapshot_hf_repo: str | None,
    log_prefix: str,
) -> SnapshotPersistResult:
    """Write a canonical snapshot or a rejected audit copy.

    Does not evict sibling ``snapshot-*`` dirs unless the health check
    passes and :meth:`SnapshotLocalStorage.write_snapshot` runs.
    """
    health = evaluate_forecaster_artifact_health(
        train_loss=train_loss,
        val_loss=val_loss,
        baseline_loss=None,
        artifact_dir=None,
    )
    if health.is_healthy:
        path = snapshot_storage.write_snapshot(
            cutoff_date=cutoff_date,
            snapshot_digest=snapshot_digest,
            model=model,
            feature_scaler=feature_scaler,
            config=config,
            metadata=metadata,
        )
        logger.info(f"{log_prefix} Saved snapshot for cutoff {cutoff_date}")
        if snapshot_hf_repo:
            try:
                snapshot_storage.upload_snapshot_to_hf(cutoff_date, snapshot_digest)
                logger.info(
                    f"{log_prefix} Uploaded snapshot {cutoff_date} to HuggingFace"
                )
            except Exception as e:
                logger.error(f"{log_prefix} Failed to upload snapshot to HF: {e}")
        return SnapshotPersistResult(
            is_canonical=True,
            failure_reasons=[],
            path=path,
        )

    rejected_metadata = {**metadata, "failure_reasons": list(health.failure_reasons)}
    path = snapshot_storage.write_rejected_snapshot(
        cutoff_date=cutoff_date,
        snapshot_digest=snapshot_digest,
        model=model,
        feature_scaler=feature_scaler,
        config=config,
        metadata=rejected_metadata,
    )
    logger.warning(
        f"{log_prefix} Snapshot for cutoff {cutoff_date} failed health check "
        f"({health.failure_reasons}); wrote rejected copy at {path}. "
        "Canonical snapshot-* dirs were not modified. Delete the rejected "
        "dir to retry this digest."
    )
    return SnapshotPersistResult(
        is_canonical=False,
        failure_reasons=list(health.failure_reasons),
        path=path,
    )

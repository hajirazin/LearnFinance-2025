"""Shared on-disk writes for canonical and rejected forecaster snapshots."""

from __future__ import annotations

import json
import logging
import pickle
import shutil
from datetime import date
from pathlib import Path
from typing import Any

import torch
from sklearn.preprocessing import StandardScaler

from brain_api.storage.forecaster_snapshots.snapshot_layout import (
    parse_hashed_snapshot_folder_name,
)

logger = logging.getLogger(__name__)

SNAPSHOT_ARTIFACT_NAMES: tuple[str, ...] = (
    "weights.pt",
    "feature_scaler.pkl",
    "config.json",
    "metadata.json",
)


def snapshot_train_val_losses(
    metadata: dict[str, Any],
) -> tuple[float, float] | None:
    """Return ``(train_loss, val_loss)`` from snapshot metadata, or None if absent."""
    metrics = metadata.get("metrics")
    if not isinstance(metrics, dict):
        return None
    train_loss = metrics.get("train_loss")
    val_loss = metrics.get("val_loss")
    if train_loss is None or val_loss is None:
        return None
    try:
        return float(train_loss), float(val_loss)
    except (TypeError, ValueError):
        return None


def write_snapshot_artifact_files(
    snapshot_dir: Path,
    *,
    model: Any,
    feature_scaler: StandardScaler,
    config: Any,
    metadata: dict[str, Any],
) -> None:
    """Write the four snapshot artifacts into ``snapshot_dir`` (created if needed)."""
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), snapshot_dir / "weights.pt")
    with open(snapshot_dir / "feature_scaler.pkl", "wb") as f:
        pickle.dump(feature_scaler, f)
    with open(snapshot_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    with open(snapshot_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def evict_sibling_hashed_snapshot_dirs(
    models_path: Path,
    cutoff_date: date,
    keep_dir: Path,
) -> None:
    """Remove other ``snapshot-{cutoff}-*`` dirs; leave ``keep_dir`` and legacy flats."""
    if not models_path.exists():
        return
    keep_resolved = keep_dir.resolve()
    for stale in models_path.glob(f"snapshot-{cutoff_date.isoformat()}-*"):
        if not stale.is_dir():
            continue
        if parse_hashed_snapshot_folder_name(stale.name) is None:
            continue
        try:
            if stale.resolve() != keep_resolved:
                shutil.rmtree(stale)
        except OSError as exc:
            logger.warning(f"Could not remove stale snapshot dir {stale}: {exc}")


def copy_snapshot_artifacts(src_path: Path, dest_dir: Path) -> None:
    """Copy the four snapshot artifact files from ``src_path`` into ``dest_dir``."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for file_name in SNAPSHOT_ARTIFACT_NAMES:
        src_file = src_path / file_name
        if src_file.exists():
            (dest_dir / file_name).write_bytes(src_file.read_bytes())

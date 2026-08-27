"""Atomic per-seed checkpoints. Resume by hashes, not HTTP job ids."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from brain_api.core.ppo_discovery.config import PPODiscoveryConfig
from brain_api.core.ppo_discovery.policy import PPODiscoveryActorCritic
from brain_api.core.ppo_discovery.schemas import PPODiscoveryError


def model_config_hash(config: PPODiscoveryConfig) -> str:
    payload = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def hash_state_dict(state_dict: Mapping[str, Any]) -> str:
    """Stable SHA-256 of a tensor state_dict (Stage-A encoder identity)."""
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key]
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def seed_checkpoint_dir(
    root: Path,
    *,
    experiment_id: str,
    snapshot_hash: str,
    config_hash: str,
) -> Path:
    return (
        Path(root)
        / "training"
        / "ppo_discovery"
        / _path_segment(experiment_id)
        / _path_segment(snapshot_hash)
        / _path_segment(config_hash)
    )


def save_seed_checkpoint(
    directory: Path,
    *,
    seed: int,
    policy: PPODiscoveryActorCritic,
    metadata: dict[str, Any] | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    weights_path = directory / f"seed-{int(seed)}.pt"
    _atomic_torch_save(policy.state_dict(), weights_path)
    payload = {
        **(metadata or {}),
        "seed": int(seed),
        "saved_at": datetime.now(UTC).isoformat(),
        "status": "complete",
        "weights_sha256": _sha256_file(weights_path),
    }
    _atomic_json_write(payload, directory / f"seed-{int(seed)}.metadata.json")


def load_seed_checkpoint(
    directory: Path,
    *,
    seed: int,
    expected: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    weights_path = directory / f"seed-{int(seed)}.pt"
    meta_path = directory / f"seed-{int(seed)}.metadata.json"
    if not weights_path.exists() or not meta_path.exists():
        return None
    metadata = json.loads(meta_path.read_text())
    if metadata.get("status") != "complete":
        return None
    if metadata.get("weights_sha256") != _sha256_file(weights_path):
        return None
    for key, value in (expected or {}).items():
        if metadata.get(key) != value:
            return None
    return {
        "state_dict": torch.load(weights_path, weights_only=True, map_location="cpu"),
        "metadata": metadata,
    }


def _path_segment(value: str) -> str:
    if not value or any(ch in value for ch in ("/", "\\")) or value in {".", ".."}:
        raise PPODiscoveryError(f"unsafe checkpoint path segment {value!r}")
    return value.replace(":", "_")


def _atomic_torch_save(obj: Any, dest: Path) -> None:
    fd, temp_path = tempfile.mkstemp(dir=dest.parent, prefix=".seed_", suffix=".tmp")
    os.close(fd)
    try:
        torch.save(obj, temp_path)
        os.replace(temp_path, dest)
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def _atomic_json_write(payload: dict[str, Any], dest: Path) -> None:
    fd, temp_path = tempfile.mkstemp(
        dir=dest.parent, prefix=".seedmeta_", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, sort_keys=True, allow_nan=False)
        os.replace(temp_path, dest)
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "hash_state_dict",
    "load_seed_checkpoint",
    "model_config_hash",
    "save_seed_checkpoint",
    "seed_checkpoint_dir",
]

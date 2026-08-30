"""Atomic per-seed checkpoints. Resume by hashes, not HTTP job ids."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from brain_api.core.ppo_discovery.config import PPODiscoveryConfig
from brain_api.core.ppo_discovery.policy import PPODiscoveryActorCritic
from brain_api.core.ppo_discovery.schemas import PPODiscoveryError

_TRAINED_STATUSES = frozenset({"trained", "complete", "validation_failed"})
_ERROR_LIMIT = 2000


def model_config_hash(config: PPODiscoveryConfig) -> str:
    payload = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def train_recipe_hash(config: PPODiscoveryConfig) -> str:
    """Config identity excluding ``seeds``. Checkpoint directory key."""
    payload = dict(config.to_dict())
    payload.pop("seeds", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


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
    recipe_hash: str,
) -> Path:
    return (
        Path(root)
        / "training"
        / "ppo_discovery"
        / _path_segment(experiment_id)
        / _path_segment(snapshot_hash)
        / _path_segment(recipe_hash)
    )


def capture_rng_state(device: torch.device) -> dict[str, Any]:
    """Python, NumPy, CPU Torch, and selected-device RNG."""
    payload: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if device.type == "cuda":
        payload["torch_cuda"] = torch.cuda.get_rng_state()
    elif device.type == "mps" and hasattr(torch.mps, "get_rng_state"):
        payload["torch_mps"] = torch.mps.get_rng_state()
    return payload


def restore_rng_state(payload: Mapping[str, Any], device: torch.device) -> None:
    """Restore RNG captured by :func:`capture_rng_state`. Do not reseed."""
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch_cpu"])
    if device.type == "cuda" and "torch_cuda" in payload:
        torch.cuda.set_rng_state(payload["torch_cuda"])
    elif (
        device.type == "mps"
        and "torch_mps" in payload
        and hasattr(torch.mps, "set_rng_state")
    ):
        torch.mps.set_rng_state(payload["torch_mps"])


def save_seed_checkpoint(
    directory: Path,
    *,
    seed: int,
    policy: PPODiscoveryActorCritic,
    metadata: dict[str, Any] | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    weights_path = directory / f"seed-{int(seed)}.pt"
    _atomic_torch_save(_cpu_state_dict(policy.state_dict()), weights_path)
    extra = dict(metadata or {})
    status = extra.pop("status", "trained")
    payload = {
        **extra,
        "seed": int(seed),
        "saved_at": datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "status": status,
        "weights_sha256": _sha256_file(weights_path),
    }
    atomic_json_write(payload, directory / f"seed-{int(seed)}.metadata.json")


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
    if metadata.get("status") not in _TRAINED_STATUSES:
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


def save_seed_partial_checkpoint(
    directory: Path,
    *,
    seed: int,
    policy: PPODiscoveryActorCritic,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    steps_done: int,
    episode_index: int,
    update_index: int,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Episode-boundary resume payload. Never load as a complete seed."""
    directory.mkdir(parents=True, exist_ok=True)
    binary = {
        "policy": _cpu_state_dict(policy.state_dict()),
        "optimizer": _cpu_optimizer_state(optimizer),
        "rng": capture_rng_state(device),
        "steps_done": int(steps_done),
        "episode_index": int(episode_index),
        "update_index": int(update_index),
    }
    weights_path = directory / f"seed-{int(seed)}.partial.pt"
    _atomic_torch_save(binary, weights_path)
    extra = dict(metadata or {})
    payload = {
        **extra,
        "seed": int(seed),
        "status": "partial",
        "steps_done": int(steps_done),
        "episode_index": int(episode_index),
        "update_index": int(update_index),
        "updated_at": datetime.now(UTC).isoformat(),
        "weights_sha256": _sha256_file(weights_path),
    }
    atomic_json_write(payload, directory / f"seed-{int(seed)}.partial.json")


def load_seed_partial_checkpoint(
    directory: Path,
    *,
    seed: int,
    expected: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    weights_path = directory / f"seed-{int(seed)}.partial.pt"
    meta_path = directory / f"seed-{int(seed)}.partial.json"
    if not weights_path.exists() or not meta_path.exists():
        return None
    metadata = json.loads(meta_path.read_text())
    if metadata.get("status") != "partial":
        return None
    if metadata.get("weights_sha256") != _sha256_file(weights_path):
        return None
    for key, value in (expected or {}).items():
        if metadata.get(key) != value:
            return None
    binary = torch.load(weights_path, weights_only=False, map_location="cpu")
    return {**binary, "metadata": metadata}


def write_seed_metadata(
    directory: Path,
    *,
    seed: int,
    payload: dict[str, Any],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    if "status" not in body:
        raise PPODiscoveryError("seed metadata must include status")
    body["seed"] = int(seed)
    body["updated_at"] = datetime.now(UTC).isoformat()
    atomic_json_write(body, directory / f"seed-{int(seed)}.metadata.json")


def bound_error_message(exc: BaseException) -> str:
    message = str(exc)
    if len(message) > _ERROR_LIMIT:
        return message[:_ERROR_LIMIT]
    return message


def _cpu_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: tensor.detach().cpu().clone() if torch.is_tensor(tensor) else tensor
        for key, tensor in state_dict.items()
    }


def _cpu_optimizer_state(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    payload = optimizer.state_dict()
    for state in payload.get("state", {}).values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.detach().cpu().clone()
    return payload


def move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


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


def atomic_json_write(payload: dict[str, Any], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
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
    "atomic_json_write",
    "bound_error_message",
    "capture_rng_state",
    "hash_state_dict",
    "load_seed_checkpoint",
    "load_seed_partial_checkpoint",
    "model_config_hash",
    "move_optimizer_state_to_device",
    "restore_rng_state",
    "save_seed_checkpoint",
    "save_seed_partial_checkpoint",
    "seed_checkpoint_dir",
    "train_recipe_hash",
    "write_seed_metadata",
]

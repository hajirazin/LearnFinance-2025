"""Durable per-seed status ledger for ppo_discovery training."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from brain_api.core.ppo_discovery.checkpoints import (
    atomic_json_write,
    bound_error_message,
    save_seed_partial_checkpoint,
    write_seed_metadata,
)
from brain_api.core.ppo_discovery.schemas import PPODiscoveryError

LEDGER_SCHEMA_VERSION = 1
LEDGER_FILENAME = "seeds_ledger.json"
logger = logging.getLogger(__name__)


def empty_seeds_ledger() -> dict[str, Any]:
    return {"schema_version": LEDGER_SCHEMA_VERSION, "seeds": {}}


def load_seeds_ledger(directory: Path) -> dict[str, Any]:
    path = Path(directory) / LEDGER_FILENAME
    if not path.exists():
        return empty_seeds_ledger()
    payload = json.loads(path.read_text())
    if int(payload.get("schema_version", 0)) != LEDGER_SCHEMA_VERSION:
        raise PPODiscoveryError(
            f"unsupported seeds_ledger schema_version {payload.get('schema_version')}"
        )
    seeds = payload.get("seeds")
    if not isinstance(seeds, dict):
        raise PPODiscoveryError("seeds_ledger.seeds must be an object")
    return {"schema_version": LEDGER_SCHEMA_VERSION, "seeds": dict(seeds)}


def write_seeds_ledger(directory: Path, ledger: dict[str, Any]) -> None:
    Path(directory).mkdir(parents=True, exist_ok=True)
    seeds = ledger.get("seeds") or {}
    for seed_id, row in seeds.items():
        if not isinstance(row, dict):
            raise PPODiscoveryError(f"ledger row for seed {seed_id} must be an object")
        if "status" not in row or "updated_at" not in row:
            raise PPODiscoveryError(
                f"ledger row for seed {seed_id} must include status and updated_at"
            )
    payload = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "seeds": seeds,
    }
    atomic_json_write(payload, Path(directory) / LEDGER_FILENAME)


def upsert_seed_row(
    ledger: dict[str, Any],
    seed: int,
    **fields: Any,
) -> dict[str, Any]:
    """Return an updated ledger with ``fields`` merged into the seed row."""
    seeds = dict(ledger.get("seeds") or {})
    row = dict(seeds.get(str(int(seed))) or {})
    row.update(fields)
    row["updated_at"] = datetime.now(UTC).isoformat()
    if "status" not in row:
        raise PPODiscoveryError("seed ledger row must include status")
    seeds[str(int(seed))] = row
    return {"schema_version": LEDGER_SCHEMA_VERSION, "seeds": seeds}


def complete_seed_rows(ledger: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for key, row in (ledger.get("seeds") or {}).items():
        if isinstance(row, dict) and row.get("status") == "complete":
            rows[int(key)] = row
    return rows


def failed_seed_ids(ledger: dict[str, Any]) -> list[int]:
    ids: list[int] = []
    for key, row in (ledger.get("seeds") or {}).items():
        if isinstance(row, dict) and row.get("status") in {
            "failed",
            "validation_failed",
        }:
            ids.append(int(key))
    return sorted(ids)


def record_episode_partial(
    directory: Path,
    *,
    seed: int,
    device,
    checkpoint_expected: Mapping[str, Any],
    counters: dict[str, int],
    ledger: dict[str, Any],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Write an episode-boundary partial and return the updated ledger."""
    counters["steps_done"] = int(payload["steps_done"])
    counters["episode_index"] = int(payload["episode_index"])
    counters["update_index"] = int(payload["update_index"])
    save_seed_partial_checkpoint(
        directory,
        seed=int(seed),
        policy=payload["policy"],
        optimizer=payload["optimizer"],
        device=device,
        steps_done=counters["steps_done"],
        episode_index=counters["episode_index"],
        update_index=counters["update_index"],
        metadata=dict(checkpoint_expected),
    )
    ledger = upsert_seed_row(
        ledger,
        int(seed),
        status="partial",
        steps_done=counters["steps_done"],
        episode_index=counters["episode_index"],
        update_index=counters["update_index"],
        device=getattr(device, "type", str(device)),
    )
    write_seeds_ledger(directory, ledger)
    return ledger


def fail_job_on_accelerator_oom(
    exc: BaseException,
    *,
    seed: int,
    device,
    directory: Path,
    ledger: dict[str, Any],
    checkpoint_expected: dict[str, Any],
    progress,
    steps_done: int = 0,
    episode_index: int = 0,
    update_index: int = 0,
) -> None:
    """Record a fatal OOM and raise. Does not start remaining seeds."""
    error = bound_error_message(exc)
    row = {
        "status": "failed",
        "fatal": True,
        "failure_scope": "job",
        "error_type": type(exc).__name__,
        "error": error,
        "device": getattr(device, "type", str(device)),
        "steps_done": int(steps_done),
        "episode_index": int(episode_index),
        "update_index": int(update_index),
        **dict(checkpoint_expected),
    }
    write_seed_metadata(directory, seed=int(seed), payload=row)
    ledger = upsert_seed_row(ledger, int(seed), **row)
    write_seeds_ledger(directory, ledger)
    progress(
        {
            "stage": "ppo_fatal",
            "seed": int(seed),
            "reason": "accelerator_out_of_memory",
            "device": getattr(device, "type", str(device)),
            "fatal": True,
        }
    )
    logger.exception("accelerator out of memory during ppo_discovery seed=%s", seed)
    raise PPODiscoveryError(
        f"accelerator_out_of_memory on seed {seed} "
        f"device={getattr(device, 'type', device)}: {error}"
    ) from exc


__all__ = [
    "LEDGER_FILENAME",
    "LEDGER_SCHEMA_VERSION",
    "complete_seed_rows",
    "empty_seeds_ledger",
    "fail_job_on_accelerator_oom",
    "failed_seed_ids",
    "load_seeds_ledger",
    "record_episode_partial",
    "upsert_seed_row",
    "write_seeds_ledger",
]

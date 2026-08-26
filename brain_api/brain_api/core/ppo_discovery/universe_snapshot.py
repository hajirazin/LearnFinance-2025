"""Freeze and persist the current ``halal_new`` roster for ppo_discovery."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from brain_api.core.ppo_discovery.config import MAX_ASSETS, UNIVERSE_NAME
from brain_api.core.ppo_discovery.schemas import (
    PPODiscoveryError,
    UniverseSnapshot,
    canonical_json_bytes,
    sha256_digest,
)
from brain_api.storage.base import DEFAULT_DATA_PATH
from brain_api.universe.halal_new import get_halal_new_symbols


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def snapshot_hash_for_symbols(
    symbols: Sequence[str], *, universe: str = UNIVERSE_NAME
) -> str:
    """Digest of the sorted roster only (stable across retrieval timestamps)."""
    sorted_symbols = tuple(sorted(symbols))
    return sha256_digest({"universe": universe, "sorted_symbols": list(sorted_symbols)})


def build_universe_snapshot(
    symbols: Sequence[str],
    *,
    retrieved_at: datetime,
    cache_path: str | None = None,
    cache_sha256: str | None = None,
    source_provenance: dict[str, Any] | None = None,
    universe: str = UNIVERSE_NAME,
) -> UniverseSnapshot:
    """Build an immutable snapshot from an already-resolved symbol list."""
    if universe != UNIVERSE_NAME:
        raise PPODiscoveryError(
            f"ppo_discovery universe must be {UNIVERSE_NAME!r}, got {universe!r}"
        )
    cleaned = [str(symbol).strip() for symbol in symbols]
    if any(not symbol for symbol in cleaned):
        raise PPODiscoveryError("universe snapshot contains a blank symbol")
    if len(set(cleaned)) != len(cleaned):
        raise PPODiscoveryError("universe snapshot contains duplicate symbols")
    sorted_symbols = tuple(sorted(cleaned))
    if len(sorted_symbols) > MAX_ASSETS:
        raise PPODiscoveryError(
            f"universe snapshot has {len(sorted_symbols)} symbols; "
            f"capacity is {MAX_ASSETS} and truncation is forbidden"
        )
    if not sorted_symbols:
        raise PPODiscoveryError("universe snapshot is empty")
    snapshot_sha256 = snapshot_hash_for_symbols(sorted_symbols, universe=universe)
    return UniverseSnapshot(
        universe=universe,
        retrieved_at=retrieved_at.astimezone(UTC).isoformat(),
        sorted_symbols=sorted_symbols,
        symbol_count=len(sorted_symbols),
        cache_path=cache_path,
        cache_sha256=cache_sha256,
        source_provenance=source_provenance or {},
        snapshot_sha256=snapshot_sha256,
    )


def persist_universe_snapshot(
    snapshot: UniverseSnapshot, *, base_path: Path | str | None = None
) -> Path:
    """Write ``<snapshot_hash>.json``; existing identical bytes are kept."""
    root = (
        Path(base_path or DEFAULT_DATA_PATH)
        / "ppo_discovery"
        / "universe"
        / "snapshots"
    )
    root.mkdir(parents=True, exist_ok=True)
    filename = snapshot.snapshot_sha256.removeprefix("sha256:") + ".json"
    path = root / filename
    encoded = canonical_json_bytes(snapshot.to_dict())
    if path.exists():
        return path
    path.write_bytes(encoded)
    return path


def resolve_universe_snapshot(
    as_of: datetime,
    *,
    symbols_resolver: Callable[[], list[str]] = get_halal_new_symbols,
    persist: bool = True,
    base_path: Path | str | None = None,
    cache_path: str | None = None,
    source_provenance: dict[str, Any] | None = None,
) -> UniverseSnapshot:
    """Resolve ``halal_new`` once and freeze the sorted roster.

    ``as_of`` is recorded as provenance. The builder is reused unchanged;
    this function never edits the monthly universe cache.
    """
    del as_of  # roster is current-snapshot by design; timestamp is provenance only.
    symbols = list(symbols_resolver())
    cache_sha256 = _file_sha256(Path(cache_path)) if cache_path else None
    snapshot = build_universe_snapshot(
        symbols,
        retrieved_at=datetime.now(UTC),
        cache_path=cache_path,
        cache_sha256=cache_sha256,
        source_provenance=source_provenance or {"builder": "get_halal_new_symbols"},
    )
    if persist:
        persist_universe_snapshot(snapshot, base_path=base_path)
    return snapshot


def load_universe_snapshot(
    snapshot_sha256: str, *, base_path: Path | str | None = None
) -> UniverseSnapshot:
    """Load a previously persisted snapshot by digest."""
    import json

    root = (
        Path(base_path or DEFAULT_DATA_PATH)
        / "ppo_discovery"
        / "universe"
        / "snapshots"
    )
    path = root / f"{snapshot_sha256.removeprefix('sha256:')}.json"
    if not path.is_file():
        raise PPODiscoveryError(f"universe snapshot {snapshot_sha256} is not on disk")
    payload = json.loads(path.read_text())
    loaded = UniverseSnapshot(
        universe=payload["universe"],
        retrieved_at=payload["retrieved_at"],
        sorted_symbols=tuple(payload["sorted_symbols"]),
        symbol_count=int(payload["symbol_count"]),
        cache_path=payload.get("cache_path"),
        cache_sha256=payload.get("cache_sha256"),
        source_provenance=payload.get("source_provenance") or {},
        snapshot_sha256=payload["snapshot_sha256"],
    )
    if loaded.snapshot_sha256 != snapshot_hash_for_symbols(loaded.sorted_symbols):
        raise PPODiscoveryError("persisted snapshot hash does not match roster")
    return loaded


__all__ = [
    "build_universe_snapshot",
    "load_universe_snapshot",
    "persist_universe_snapshot",
    "resolve_universe_snapshot",
    "snapshot_hash_for_symbols",
]

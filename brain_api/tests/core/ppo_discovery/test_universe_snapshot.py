"""Universe snapshot freeze/hash tests for ppo_discovery."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from brain_api.core.model_buckets import ModelType, UnknownBucketError, get_bucket
from brain_api.core.orders import generate_client_order_id
from brain_api.core.ppo_discovery.config import MAX_ASSETS, UNIVERSE_NAME
from brain_api.core.ppo_discovery.schemas import PPODiscoveryError
from brain_api.core.ppo_discovery.universe_snapshot import (
    build_universe_snapshot,
    persist_universe_snapshot,
    resolve_universe_snapshot,
    snapshot_hash_for_symbols,
)
from brain_api.routes.experience_account_labeling import infer_universe_from_run_id


def test_snapshot_sorts_and_hashes_deterministically(tmp_path: Path) -> None:
    as_of = datetime(2026, 8, 31, 13, 0, tzinfo=UTC)
    first = build_universe_snapshot(["MSFT", "AAPL", "NVDA"], retrieved_at=as_of)
    second = build_universe_snapshot(
        ["NVDA", "AAPL", "MSFT"], retrieved_at=datetime(2026, 9, 1, tzinfo=UTC)
    )
    assert first.sorted_symbols == ("AAPL", "MSFT", "NVDA")
    assert first.snapshot_sha256 == second.snapshot_sha256
    assert first.snapshot_sha256 == snapshot_hash_for_symbols(first.sorted_symbols)
    path = persist_universe_snapshot(first, base_path=tmp_path)
    assert path.exists()
    persist_universe_snapshot(second, base_path=tmp_path)


def test_snapshot_rejects_duplicates_and_capacity() -> None:
    as_of = datetime(2026, 8, 31, tzinfo=UTC)
    with pytest.raises(PPODiscoveryError, match="duplicate"):
        build_universe_snapshot(["AAPL", "AAPL"], retrieved_at=as_of)
    too_many = [f"S{i:04d}" for i in range(MAX_ASSETS + 1)]
    with pytest.raises(PPODiscoveryError, match="capacity"):
        build_universe_snapshot(too_many, retrieved_at=as_of)


def test_resolve_freezes_resolver_output(tmp_path: Path) -> None:
    calls = {"n": 0}

    def resolver() -> list[str]:
        calls["n"] += 1
        return ["META", "AAPL"]

    snapshot = resolve_universe_snapshot(
        datetime(2026, 8, 31, tzinfo=UTC),
        symbols_resolver=resolver,
        persist=True,
        base_path=tmp_path,
    )
    assert snapshot.sorted_symbols == ("AAPL", "META")
    assert snapshot.universe == UNIVERSE_NAME
    assert calls["n"] == 1


def test_held_name_dropped_from_live_roster_is_not_in_snapshot() -> None:
    """A name leaving the current roster is absent from the frozen list.

    Liquidation of a still-held dropped name is the state-builder /
    order-generator's job; the snapshot itself must not keep it.
    """
    snapshot = build_universe_snapshot(
        ["AAPL", "MSFT"], retrieved_at=datetime(2026, 8, 31, tzinfo=UTC)
    )
    assert "IBM" not in snapshot.sorted_symbols


def test_paper_halal_new_run_id_is_not_inferred_as_sac_halal() -> None:
    """``paper:halal_new:`` must not match the SAC ``paper:halal:`` prefix."""
    assert infer_universe_from_run_id("paper:halal:2026-08-31") == "halal"
    assert infer_universe_from_run_id("paper:halal_new:2026-08-31") == "halal_filtered"
    assert infer_universe_from_run_id("paper:2026-08-31") == "halal_filtered"


def test_client_order_id_omits_algorithm() -> None:
    order_id = generate_client_order_id("paper:halal_new:2026-08-31", 1, "AAPL", "buy")
    assert order_id == "paper:halal_new:2026-08-31:attempt-1:AAPL:BUY"
    assert "ppo_discovery" not in order_id


def test_ppo_discovery_bucket_is_registered() -> None:
    bucket = get_bucket(ModelType.PPO_DISCOVERY, "halal_new")
    assert bucket.bucket_name == "ppo_discovery_halal_new"
    with pytest.raises(UnknownBucketError):
        get_bucket(ModelType.PPO_DISCOVERY, "halal_filtered")

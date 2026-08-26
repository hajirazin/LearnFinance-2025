"""Walk-forward split tests for ppo_discovery."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from brain_api.core.ppo_discovery.config import PPODiscoveryConfig
from brain_api.core.ppo_discovery.environment import WeeklyTransition
from brain_api.core.ppo_discovery.pipeline import run_ppo_discovery_training
from brain_api.core.ppo_discovery.schemas import PPODiscoveryError
from brain_api.core.ppo_discovery.splits import (
    resolve_experiment_variant,
    split_walk_forward,
)
from brain_api.core.ppo_discovery.universe_snapshot import build_universe_snapshot


def _weeks(n: int) -> list[WeeklyTransition]:
    start = datetime(2024, 1, 1, 9, 0, tzinfo=UTC)
    rows = []
    for index in range(n):
        cutoff = start + timedelta(days=7 * index)
        rows.append(
            WeeklyTransition(
                cutoff=cutoff,
                rebalance_session=cutoff,
                next_rebalance_session=cutoff + timedelta(days=7),
                news_by_symbol={},
                p_calm=0.0,
                p_stress=0.0,
            )
        )
    return rows


def test_full_split_purges_21_days() -> None:
    train, val, test = split_walk_forward(_weeks(40), experiment_variant="full")
    assert train and val and test
    assert train[-1].cutoff <= val[0].cutoff - timedelta(days=21)
    assert val[-1].cutoff <= test[0].cutoff - timedelta(days=21)
    assert train[-1].cutoff < val[0].cutoff < test[0].cutoff


def test_full_split_raises_when_purge_empties_train() -> None:
    with pytest.raises(PPODiscoveryError, match="purge"):
        split_walk_forward(_weeks(6), experiment_variant="full")


def test_diagnostic_split_is_disjoint_without_purge() -> None:
    train, val, test = split_walk_forward(_weeks(20), experiment_variant="diagnostic")
    assert {w.cutoff for w in train}.isdisjoint({w.cutoff for w in val})
    assert {w.cutoff for w in val}.isdisjoint({w.cutoff for w in test})


def test_timestep_override_is_diagnostic_not_full() -> None:
    locked = PPODiscoveryConfig()
    assert resolve_experiment_variant(locked) == "full"
    reduced = PPODiscoveryConfig(total_timesteps=1, seeds=(42,))
    assert resolve_experiment_variant(reduced) == "diagnostic"


def test_pipeline_rejects_full_variant_for_unlocked_config() -> None:
    snapshot = build_universe_snapshot(
        ["AAPL", "MSFT"], retrieved_at=datetime(2026, 8, 24, tzinfo=UTC)
    )
    with pytest.raises(PPODiscoveryError, match="10_000 timesteps"):
        run_ppo_discovery_training(
            snapshot,
            config=PPODiscoveryConfig(total_timesteps=4, seeds=(42,)),
            storage=object(),  # type: ignore[arg-type]
            end_date=datetime(2026, 8, 24, tzinfo=UTC).date(),
            experiment_id="diag",
            experiment_variant="full",
        )

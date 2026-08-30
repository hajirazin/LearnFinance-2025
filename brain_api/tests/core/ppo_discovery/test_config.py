"""Locked PPO Discovery broker-cost and capital contract."""

from __future__ import annotations

import pytest

from brain_api.core.portfolio_rl.broker_costs import IBKRSingaporeCostConfig
from brain_api.core.ppo_discovery.config import (
    EXPERIMENT_SEEDS,
    PPO_DISCOVERY_BROKER_COST_MODEL,
    PPO_DISCOVERY_TRAINING_NAV_USD,
    PPODiscoveryConfig,
)


def test_ppo_cost_contract_is_fixed_to_ibkr_and_ten_thousand_dollars() -> None:
    config = PPODiscoveryConfig()

    assert config.training_nav_usd == 10_000.0
    assert PPO_DISCOVERY_TRAINING_NAV_USD == 10_000.0
    assert PPO_DISCOVERY_BROKER_COST_MODEL == "ibkr_sg_tiered"
    assert config.to_dict()["broker_cost_config"] == (
        IBKRSingaporeCostConfig.default().to_dict()
    )


def test_training_nav_is_not_a_constructor_parameter() -> None:
    with pytest.raises(TypeError, match="training_nav_usd"):
        PPODiscoveryConfig(training_nav_usd=100_000.0)  # type: ignore[call-arg]


def test_locked_cost_contract_round_trips() -> None:
    config = PPODiscoveryConfig(dropout=0.0, seeds=(42,))

    restored = PPODiscoveryConfig.from_dict(config.to_dict())

    assert restored.to_dict() == config.to_dict()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("broker_cost_model", "alpaca_us", "broker_cost_model"),
        ("training_nav_usd", 100_000.0, "training_nav_usd"),
        ("broker_cost_config", {}, "broker_cost_config"),
    ],
)
def test_deserialization_rejects_unlocked_cost_contract(
    field: str, value: object, message: str
) -> None:
    payload = PPODiscoveryConfig().to_dict()
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        PPODiscoveryConfig.from_dict(payload)


def test_deserialization_rejects_missing_cost_contract() -> None:
    payload = PPODiscoveryConfig().to_dict()
    payload.pop("broker_cost_model")

    with pytest.raises(ValueError, match="broker_cost_model"):
        PPODiscoveryConfig.from_dict(payload)


def test_default_experiment_seeds_are_the_full_protocol() -> None:
    assert EXPERIMENT_SEEDS == (42, 123, 2026)
    assert PPODiscoveryConfig().seeds == (42, 123, 2026)
    assert PPODiscoveryConfig().ppo_microbatch_size == 8
    assert PPODiscoveryConfig().minibatch_size == 32


def test_recipe_hash_ignores_only_seeds() -> None:
    from brain_api.core.ppo_discovery.checkpoints import (
        model_config_hash,
        train_recipe_hash,
    )

    base = PPODiscoveryConfig()
    other_seeds = PPODiscoveryConfig(seeds=(42,))
    other_dropout = PPODiscoveryConfig(dropout=0.0)
    assert train_recipe_hash(base) == train_recipe_hash(other_seeds)
    assert train_recipe_hash(base) != train_recipe_hash(other_dropout)
    assert model_config_hash(base) != model_config_hash(other_seeds)


def test_microbatch_must_divide_minibatch() -> None:
    with pytest.raises(ValueError, match="divisible"):
        PPODiscoveryConfig(minibatch_size=32, ppo_microbatch_size=3)
    with pytest.raises(ValueError, match="ppo_microbatch_size"):
        PPODiscoveryConfig(minibatch_size=8, ppo_microbatch_size=16)

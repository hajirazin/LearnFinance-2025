"""Locked PPO Discovery broker-cost and capital contract."""

from __future__ import annotations

import pytest

from brain_api.core.portfolio_rl.broker_costs import IBKRSingaporeCostConfig
from brain_api.core.ppo_discovery.config import (
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

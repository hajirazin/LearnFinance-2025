"""GAE, freeze-20, and tiny synthetic PPO overfit tests."""

from __future__ import annotations

import numpy as np

from brain_api.core.ppo_discovery.config import PPODiscoveryConfig
from brain_api.core.ppo_discovery.policy import PPODiscoveryActorCritic
from brain_api.core.ppo_discovery.pretraining import next_week_open_log_return, rank_ic
from brain_api.core.ppo_discovery.rollout import compute_gae, normalize_advantages
from brain_api.core.ppo_discovery.state_builder import build_ppo_discovery_state
from brain_api.core.ppo_discovery.trainer import train_ppo_discovery
from tests.core.ppo_discovery.test_state_builder import _request


def test_next_week_open_target() -> None:
    assert next_week_open_log_return(100.0, 110.0) == np.log(1.1)


def test_gae_and_advantage_normalization() -> None:
    adv, _ret = compute_gae(
        [1.0, 1.0],
        [0.0, 0.0],
        [False, True],
        gamma=0.97,
        gae_lambda=0.95,
    )
    assert len(adv) == 2
    norm = normalize_advantages(adv)
    assert abs(norm.mean()) < 1e-8


def test_tiny_synthetic_ppo_runs() -> None:
    from brain_api.core.ppo_discovery.rollout import collect_rollout

    state = build_ppo_discovery_state(_request())
    config = PPODiscoveryConfig(
        total_timesteps=8,
        rollout_length=4,
        minibatch_size=2,
        ppo_epochs=1,
        freeze_encoder_updates=20,
        dropout=0.0,
    )
    policy = PPODiscoveryActorCritic(config)

    def episode(current):
        return collect_rollout(
            current, [state, state], [0.01, 0.0], [False, True], config=config
        )

    metrics = train_ppo_discovery(policy, episode, config=config, seed=42)
    assert metrics["timesteps"] >= 8
    for parameter in policy.temporal.parameters():
        assert parameter.requires_grad is False


def test_rank_ic_perfect() -> None:
    assert rank_ic(np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0])) == 1.0

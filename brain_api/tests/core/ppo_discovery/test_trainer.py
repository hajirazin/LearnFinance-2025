"""GAE, freeze-20, and tiny synthetic PPO overfit tests."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from brain_api.core.ppo_discovery.config import PPODiscoveryConfig
from brain_api.core.ppo_discovery.policy import PPODiscoveryActorCritic
from brain_api.core.ppo_discovery.pretraining import next_week_open_log_return, rank_ic
from brain_api.core.ppo_discovery.rollout import compute_gae, normalize_advantages
from brain_api.core.ppo_discovery.schemas import PPODiscoveryError
from brain_api.core.ppo_discovery.state_builder import build_ppo_discovery_state
from brain_api.core.ppo_discovery.trainer import (
    _assert_gradient_devices,
    train_ppo_discovery,
)
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
        ppo_microbatch_size=2,
        ppo_epochs=1,
        freeze_encoder_updates=20,
        dropout=0.0,
    )
    policy = PPODiscoveryActorCritic(config)

    def episode(current, cache=None):
        return collect_rollout(
            current, [state, state], [0.01, 0.0], [False, True], config=config
        )

    metrics = train_ppo_discovery(policy, episode, config=config, seed=42)
    assert metrics["timesteps"] >= 8
    for parameter in policy.temporal.parameters():
        assert parameter.requires_grad is False


def test_rank_ic_average_ties() -> None:
    tied = rank_ic(np.array([1.0, 1.0, 2.0]), np.array([10.0, 10.0, 30.0]))
    assert tied == pytest.approx(1.0)


def test_rollout_length_chunks_ppo_updates(monkeypatch) -> None:
    from brain_api.core.ppo_discovery import trainer as trainer_mod
    from brain_api.core.ppo_discovery.rollout import collect_rollout

    chunk_sizes: list[int] = []

    def fake_update(
        policy, steps, optimizer, *, config, update_index, last_value=0.0, **kwargs
    ):
        del policy, optimizer, config, last_value, kwargs
        chunk_sizes.append(len(steps))
        return {"ppo_loss": 0.0, "update_index": float(update_index)}

    monkeypatch.setattr(trainer_mod, "ppo_update", fake_update)
    state = build_ppo_discovery_state(_request())
    config = PPODiscoveryConfig(
        total_timesteps=8,
        rollout_length=4,
        minibatch_size=2,
        ppo_microbatch_size=2,
        ppo_epochs=1,
        dropout=0.0,
    )
    policy = PPODiscoveryActorCritic(config)

    def episode(current, cache=None):
        rewards = [0.01] * 8
        dones = [False] * 7 + [True]
        return collect_rollout(current, [state] * 8, rewards, dones, config=config)

    metrics = trainer_mod.train_ppo_discovery(policy, episode, config=config, seed=0)
    assert chunk_sizes == [4, 4]
    assert metrics["timesteps"] == 8.0


def test_entropy_coefs_are_applied_to_matching_heads() -> None:
    import inspect

    from brain_api.core.ppo_discovery.trainer import ppo_update

    source = inspect.getsource(ppo_update)
    assert "count_entropy_coef" in source
    assert "selection_entropy_coef" in source
    assert "count_entropy_coef + selection_entropy_coef" not in source
    assert "n_entropy_draws" not in source


def test_chunked_gae_bootstraps_next_state_value(monkeypatch) -> None:
    from dataclasses import replace

    from brain_api.core.ppo_discovery import trainer as trainer_mod
    from brain_api.core.ppo_discovery.rollout import collect_rollout

    last_values: list[float] = []

    def fake_update(
        policy, steps, optimizer, *, config, update_index, last_value=0.0, **kwargs
    ):
        del policy, optimizer, config, steps, kwargs
        last_values.append(float(last_value))
        return {"ppo_loss": 0.0, "update_index": float(update_index)}

    monkeypatch.setattr(trainer_mod, "ppo_update", fake_update)
    state = build_ppo_discovery_state(_request())
    config = PPODiscoveryConfig(
        total_timesteps=8,
        rollout_length=4,
        minibatch_size=2,
        ppo_microbatch_size=2,
        ppo_epochs=1,
        dropout=0.0,
    )
    policy = PPODiscoveryActorCritic(config)

    def episode(current, cache=None):
        rewards = [0.01] * 8
        dones = [False] * 7 + [True]
        steps = collect_rollout(current, [state] * 8, rewards, dones, config=config)
        return [
            replace(step, value=float(index + 1)) for index, step in enumerate(steps)
        ]

    trainer_mod.train_ppo_discovery(policy, episode, config=config, seed=0)
    assert last_values == [5.0, 0.0]


def test_four_microbatches_take_one_optimizer_step() -> None:
    from brain_api.core.ppo_discovery.rollout import collect_rollout
    from brain_api.core.ppo_discovery.trainer import ppo_update
    from tests.core.ppo_discovery.test_state_builder import _request

    state = build_ppo_discovery_state(_request())
    config = PPODiscoveryConfig(
        total_timesteps=32,
        rollout_length=32,
        minibatch_size=32,
        ppo_microbatch_size=8,
        ppo_epochs=1,
        dropout=0.0,
        freeze_encoder_updates=20,
    )
    policy = PPODiscoveryActorCritic(config)
    policy.eval()
    steps = collect_rollout(
        policy,
        [state] * 32,
        [0.01] * 32,
        [False] * 31 + [True],
        config=config,
    )
    encodes = {"n": 0}
    original_encode = policy.encode

    def wrapped(*args, **kwargs):
        encodes["n"] += 1
        return original_encode(*args, **kwargs)

    policy.encode = wrapped  # type: ignore[method-assign]
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    ppo_update(
        policy,
        steps,
        optimizer,
        config=config,
        update_index=0,
    )
    assert encodes["n"] == 4


def test_frozen_train_episode_receives_temporal_cache() -> None:
    from brain_api.core.ppo_discovery.rollout import collect_rollout
    from brain_api.core.ppo_discovery.temporal_cache import FrozenTemporalEmbeddingCache

    state = build_ppo_discovery_state(_request())
    config = PPODiscoveryConfig(
        total_timesteps=8,
        rollout_length=4,
        minibatch_size=2,
        ppo_microbatch_size=2,
        ppo_epochs=1,
        freeze_encoder_updates=20,
        dropout=0.0,
    )
    policy = PPODiscoveryActorCritic(config)
    seen: list[object] = []

    def episode(current, cache):
        seen.append(cache)
        return collect_rollout(
            current, [state, state], [0.01, 0.0], [False, True], config=config
        )

    train_ppo_discovery(policy, episode, config=config, seed=0)
    assert seen
    assert all(isinstance(cache, FrozenTemporalEmbeddingCache) for cache in seen)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is not available"
)
def test_tiny_synthetic_ppo_update_runs_on_mps() -> None:
    from brain_api.core.ppo_discovery.rollout import collect_rollout

    assert "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ
    state = build_ppo_discovery_state(_request())
    config = PPODiscoveryConfig(
        total_timesteps=2,
        rollout_length=2,
        minibatch_size=2,
        ppo_microbatch_size=2,
        ppo_epochs=1,
        freeze_encoder_updates=20,
        dropout=0.0,
    )
    policy = PPODiscoveryActorCritic(config).to("mps")

    def episode(current, cache=None):
        return collect_rollout(
            current, [state, state], [0.01, 0.0], [False, True], config=config
        )

    metrics = train_ppo_discovery(
        policy, episode, config=config, seed=42, device=torch.device("mps")
    )
    assert metrics["timesteps"] >= 2
    for parameter in policy.parameters():
        assert parameter.device.type == "mps"
        if parameter.grad is not None:
            assert parameter.grad.device.type == "mps"


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is not available"
)
def test_production_shaped_first_ppo_update_runs_on_mps() -> None:
    """Live first-update batching on MPS with K>0 cash/Dirichlet terms.

    Uses production ``rollout_length`` / ``minibatch_size`` /
    ``ppo_microbatch_size``. This is not a reconstruction of the
    WatchFiles-interrupted job; a control run with ``Beta.log_prob`` /
    ``Dirichlet.log_prob`` also completed this update on PyTorch 2.9.1 MPS.
    """
    from brain_api.core.ppo_discovery.rollout import collect_rollout

    assert "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ
    state = build_ppo_discovery_state(_request())
    config = PPODiscoveryConfig(
        total_timesteps=52,
        rollout_length=52,
        minibatch_size=32,
        ppo_microbatch_size=8,
        ppo_epochs=1,
        freeze_encoder_updates=20,
        dropout=0.0,
    )
    policy = PPODiscoveryActorCritic(config).to("mps")

    def episode(current, cache=None):
        horizon = config.rollout_length
        return collect_rollout(
            current,
            [state] * horizon,
            [0.01] * horizon,
            [False] * (horizon - 1) + [True],
            config=config,
        )

    metrics = train_ppo_discovery(
        policy, episode, config=config, seed=42, device=torch.device("mps")
    )
    assert metrics["timesteps"] >= 52
    assert metrics["mean_k"] > 0
    for parameter in policy.parameters():
        assert parameter.device.type == "mps"
        if parameter.grad is not None:
            assert parameter.grad.device.type == "mps"


def test_assert_gradient_devices_passes_when_aligned() -> None:
    policy = PPODiscoveryActorCritic(PPODiscoveryConfig(dropout=0.0))
    loss = policy.value_head.weight.sum()
    loss.backward()
    _assert_gradient_devices(policy, torch.device("cpu"))


def test_assert_gradient_devices_names_mismatched_parameter() -> None:
    class _MismatchPolicy:
        def named_parameters(self):
            parameter = type("Param", (), {})()
            parameter.device = torch.device("mps")
            parameter.grad = torch.zeros(1)
            yield "value_head.weight", parameter

    with pytest.raises(PPODiscoveryError, match=r"value_head\.weight"):
        _assert_gradient_devices(_MismatchPolicy(), torch.device("mps"))

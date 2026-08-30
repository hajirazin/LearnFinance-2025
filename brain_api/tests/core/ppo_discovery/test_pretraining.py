"""Stage-A validation must disable dropout."""

from __future__ import annotations

import numpy as np
import torch

from brain_api.core.ppo_discovery.config import PPODiscoveryConfig
from brain_api.core.ppo_discovery.policy import PPODiscoveryActorCritic
from brain_api.core.ppo_discovery.pretraining import pretrain_temporal_encoder


def test_pretrain_validation_switches_to_eval_mode() -> None:
    config = PPODiscoveryConfig(
        dropout=0.5,
        pretrain_max_epochs=1,
        pretrain_patience=1,
        pretrain_batch_size=4,
    )
    policy = PPODiscoveryActorCritic(config)
    original_train = policy.train
    original_eval = policy.eval
    modes: list[bool] = []

    def train_spy(mode: bool = True):
        result = original_train(mode)
        modes.append(policy.training)
        return result

    def eval_spy():
        result = original_eval()
        modes.append(policy.training)
        return result

    policy.train = train_spy  # type: ignore[method-assign]
    policy.eval = eval_spy  # type: ignore[method-assign]
    histories = [np.zeros((2, 250, 4), dtype=np.float64) for _ in range(4)]
    targets = [np.array([0.01, 0.02], dtype=np.float64) for _ in range(4)]
    pretrain_temporal_encoder(policy, histories, targets, config=config, seed=0)
    assert False in modes


def test_seeded_init_then_pretrain_is_independent_of_leftover_rng() -> None:
    from brain_api.core.ppo_discovery.checkpoints import hash_state_dict

    config = PPODiscoveryConfig(
        dropout=0.0,
        pretrain_max_epochs=1,
        pretrain_patience=1,
        pretrain_batch_size=4,
        seeds=(42, 123, 2026),
    )
    histories = [np.zeros((2, 250, 4), dtype=np.float64) for _ in range(4)]
    targets = [np.array([0.01, 0.02], dtype=np.float64) for _ in range(4)]

    def _build(leftover: int | None):
        if leftover is not None:
            torch.manual_seed(leftover)
            np.random.seed(leftover)
            PPODiscoveryActorCritic(config)
        torch.manual_seed(config.seeds[0])
        np.random.seed(config.seeds[0])
        policy = PPODiscoveryActorCritic(config)
        pretrain_temporal_encoder(
            policy, histories, targets, config=config, seed=config.seeds[0]
        )
        return policy

    first = _build(None)
    second = _build(None)
    leftover = _build(999)
    assert hash_state_dict(first.temporal.state_dict()) == hash_state_dict(
        second.temporal.state_dict()
    )
    assert hash_state_dict(first.temporal.state_dict()) == hash_state_dict(
        leftover.temporal.state_dict()
    )
    for name in ("count_head", "selection_head", "cash_head", "value_head"):
        assert torch.equal(
            getattr(first, name).weight.detach(), getattr(second, name).weight.detach()
        )
        assert torch.equal(
            getattr(first, name).weight.detach(),
            getattr(leftover, name).weight.detach(),
        )

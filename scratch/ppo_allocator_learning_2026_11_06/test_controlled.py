import numpy as np
import pytest
import torch

from controlled_policy import AllocationContext, ConditionalAllocationPolicy
from controlled_rewards import feasible_comparator

from brain_api.core.ppo_discovery.config import PPODiscoveryConfig


def test_known_unequal_allocation_beats_equal_under_actual_reward():
    comparator = feasible_comparator()
    assert comparator["best"]["winner_weight"] > 0.30
    assert comparator["best"]["reward"] - comparator["equal"]["reward"] > 0.01


def test_likelihood_reconstruction_and_gradient_credit():
    torch.manual_seed(42)
    policy = ConditionalAllocationPolicy(PPODiscoveryConfig(), "head").eval()
    contexts = [AllocationContext(0) for _ in range(2048)]
    actions, _, old_logs = policy.sample_batch(contexts)
    logs, _, _, _ = policy.evaluate_actions(contexts, actions)
    assert np.allclose(logs.detach().numpy(), old_logs, atol=1e-5)
    q_before, _ = policy.deterministic_batch(contexts[:1])
    advantage = torch.tensor([a.dirichlet_weights[0] for a in actions])
    optimizer = torch.optim.SGD(policy.allocation_head.parameters(), lr=0.1)
    loss = -(logs * (advantage - advantage.mean())).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    q_after, _ = policy.deterministic_batch(contexts[:1])
    assert q_after[0, 0] > q_before[0, 0]


def test_hidden_signal_is_unidentifiable_and_cash_is_fixed():
    policy = ConditionalAllocationPolicy(PPODiscoveryConfig(), "set").eval()
    contexts = [AllocationContext(0, False), AllocationContext(7, False)]
    q, _ = policy.deterministic_batch(contexts)
    assert np.allclose(q, 0.1, atol=1e-6)
    actions, _, _ = policy.sample_batch(contexts)
    assert all(a.percentage_weights["CASH"] == pytest.approx(0.1) for a in actions)
    assert all(
        sum(a.percentage_weights.values()) == pytest.approx(1, abs=1e-6)
        for a in actions
    )

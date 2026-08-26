"""Policy math tests: Plackett-Luce, cash floor, logp replay, padding."""

from __future__ import annotations

import numpy as np
import torch

from brain_api.core.ppo_discovery.config import (
    CASH_FLOOR,
    MAX_ASSETS,
    PPODiscoveryConfig,
)
from brain_api.core.ppo_discovery.distributions import (
    clamp_concentration,
    sample_factored_action,
)
from brain_api.core.ppo_discovery.policy import PPODiscoveryActorCritic
from brain_api.core.ppo_discovery.state_builder import build_ppo_discovery_state
from brain_api.core.ppo_discovery.temporal_encoder import PPODiscoveryTemporalEncoder
from tests.core.ppo_discovery.test_state_builder import _request


def _tiny_policy() -> PPODiscoveryActorCritic:
    torch.manual_seed(0)
    return PPODiscoveryActorCritic(PPODiscoveryConfig(dropout=0.0))


def test_patch_shapes() -> None:
    encoder = PPODiscoveryTemporalEncoder()
    history = torch.zeros(2, 12, 250, 4)
    out = encoder(history)
    assert out.shape == (2, 12, 64)


def test_k0_is_all_cash() -> None:
    logits = torch.zeros(16)
    logits[0] = 10.0
    selection = torch.zeros(MAX_ASSETS)
    mask = torch.zeros(MAX_ASSETS, dtype=torch.bool)
    mask[:12] = True
    symbols = tuple([f"S{i:02d}" for i in range(MAX_ASSETS)])
    action = sample_factored_action(
        count_logits=logits,
        selection_logits=selection,
        cash_raw=torch.tensor([0.0, 0.0]),
        allocation_raw=torch.zeros(MAX_ASSETS),
        asset_mask=mask,
        symbols=symbols,
    )
    assert action.k == 0
    assert action.percentage_weights == {"CASH": 1.0}
    assert action.log_p_cash == 0.0
    assert action.log_p_dirichlet == 0.0


def test_unique_plackett_luce_and_cash_floor() -> None:
    torch.manual_seed(1)
    logits = torch.zeros(16)
    logits[3] = 20.0
    selection = torch.arange(MAX_ASSETS, dtype=torch.float32)
    mask = torch.zeros(MAX_ASSETS, dtype=torch.bool)
    mask[:12] = True
    symbols = tuple([f"S{i:02d}" for i in range(MAX_ASSETS)])
    action = sample_factored_action(
        count_logits=logits,
        selection_logits=selection,
        cash_raw=torch.tensor([2.0, 2.0]),
        allocation_raw=torch.ones(MAX_ASSETS),
        asset_mask=mask,
        symbols=symbols,
    )
    assert action.k == 3
    assert len(set(action.selection_order)) == 3
    assert action.percentage_weights["CASH"] >= CASH_FLOOR - 1e-6
    stock_sum = sum(w for s, w in action.percentage_weights.items() if s != "CASH")
    assert abs(stock_sum + action.percentage_weights["CASH"] - 1.0) < 1e-6


def test_concentrations_in_range() -> None:
    raw = torch.tensor([-10.0, 0.0, 100.0])
    clamped = clamp_concentration(raw)
    assert torch.all(clamped >= 1.0)
    assert torch.all(clamped <= 50.0)


def test_logp_replay_after_optimizer_step() -> None:
    state = build_ppo_discovery_state(_request())
    policy = _tiny_policy()
    torch.manual_seed(2)
    action = policy.sample_action(state)
    stored = float(action.log_p_total)
    recomputed = float(policy.log_prob(state, action).item())
    assert abs(stored - recomputed) < 1e-6
    opt = torch.optim.Adam(policy.parameters(), lr=1e-4)
    loss = policy.log_prob(state, action)
    loss.backward()
    opt.step()
    after_step = float(policy.log_prob(state, action).item())
    assert np.isfinite(after_step)
    policy.zero_grad()
    first = float(policy.log_prob(state, action).item())
    second = float(policy.log_prob(state, action).item())
    assert abs(first - second) < 1e-6


def test_padding_invariance_of_selection_logits() -> None:
    policy = _tiny_policy()
    state = build_ppo_discovery_state(_request())
    device = next(policy.parameters()).device
    from brain_api.core.ppo_discovery.policy import tensors_from_state

    history, features, globals_, mask = tensors_from_state(state, device)
    encoded, pooled = policy.encode(
        history.unsqueeze(0),
        features.unsqueeze(0),
        globals_.unsqueeze(0),
        mask.unsqueeze(0),
    )
    _, logits, _ = policy.heads(encoded, pooled)
    padded = history.clone()
    padded[-1] = 123.0
    encoded2, pooled2 = policy.encode(
        padded.unsqueeze(0),
        features.unsqueeze(0),
        globals_.unsqueeze(0),
        mask.unsqueeze(0),
    )
    _, logits2, _ = policy.heads(encoded2, pooled2)
    eligible = mask
    assert torch.allclose(logits[0][eligible], logits2[0][eligible], atol=1e-5)


def test_deterministic_inference_simplex() -> None:
    policy = _tiny_policy()
    state = build_ppo_discovery_state(_request())
    weights = policy.infer_weights(state)
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert "CASH" in weights

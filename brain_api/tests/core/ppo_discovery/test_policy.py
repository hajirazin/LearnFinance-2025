"""Policy math tests: Plackett-Luce, cash floor, logp replay, padding."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from brain_api.core.ppo_discovery.config import (
    CASH_FLOOR,
    MAX_ASSETS,
    PPODiscoveryConfig,
)
from brain_api.core.ppo_discovery.distributions import (
    clamp_concentration,
    recompute_action_log_prob_tensors,
    sample_cash_and_weights,
    sample_factored_action,
)
from brain_api.core.ppo_discovery.policy import (
    PPODiscoveryActorCritic,
    tensors_from_state,
)
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


def test_selection_order_follows_selection_logits_not_alpha() -> None:
    policy = _tiny_policy()
    state = build_ppo_discovery_state(_request())
    _weights, order = policy.infer_decision(state)
    if not order:
        return
    device = next(policy.parameters()).device
    history, features, globals_, mask = tensors_from_state(state, device)
    encoded, pooled = policy.encode(
        history.unsqueeze(0),
        features.unsqueeze(0),
        globals_.unsqueeze(0),
        mask.unsqueeze(0),
    )
    _count, selection_logits, _ = policy.heads(encoded, pooled)
    valid_indices = [index for index, flag in enumerate(mask.tolist()) if flag]
    ranked = sorted(
        valid_indices,
        key=lambda index: (
            -float(selection_logits[0, index].item()),
            state.symbols[index],
        ),
    )
    expected = tuple(state.symbols[index] for index in ranked[: len(order)])
    assert order == expected


def test_selection_entropy_sums_all_plackett_luce_draws() -> None:
    from brain_api.core.ppo_discovery.distributions import count_and_selection_entropy

    count_logits = torch.zeros(16)
    selection_logits = torch.zeros(MAX_ASSETS)
    selection_logits[:4] = torch.tensor([3.0, 2.0, 1.0, 0.5])
    mask = torch.zeros(MAX_ASSETS, dtype=torch.bool)
    mask[:4] = True
    _, first_only = count_and_selection_entropy(
        count_logits=count_logits,
        selection_logits=selection_logits,
        asset_mask=mask,
        selection_indices=(0,),
        k=1,
    )
    _, all_draws = count_and_selection_entropy(
        count_logits=count_logits,
        selection_logits=selection_logits,
        asset_mask=mask,
        selection_indices=(0, 1, 2),
        k=3,
    )
    _, zero = count_and_selection_entropy(
        count_logits=count_logits,
        selection_logits=selection_logits,
        asset_mask=mask,
        selection_indices=(),
        k=0,
    )
    assert float(all_draws) > float(first_only)
    assert float(zero) == 0.0


def test_fused_sample_is_one_encode_and_matches_heads() -> None:
    state = build_ppo_discovery_state(_request())
    policy = _tiny_policy()
    torch.manual_seed(4)
    encodes = {"n": 0}
    original = policy.encode

    def wrapped(*args, **kwargs):
        encodes["n"] += 1
        return original(*args, **kwargs)

    policy.encode = wrapped  # type: ignore[method-assign]
    action, value, log_p = policy.sample_action_value_log_prob(state)
    assert encodes["n"] == 1
    assert log_p == pytest.approx(action.log_p_total)
    policy.encode = original  # type: ignore[method-assign]
    torch.manual_seed(4)
    action2 = policy.sample_action(state)
    assert action.k == action2.k
    assert abs(value - float(policy.value(state).item())) < 1e-5


def test_evaluate_actions_eight_state_microbatch_matches_scalar() -> None:
    state = build_ppo_discovery_state(_request())
    policy = _tiny_policy()
    torch.manual_seed(5)
    actions = [policy.sample_action(state) for _ in range(8)]
    states = [state] * 8
    encodes = {"n": 0}
    original = policy.encode

    def wrapped(*args, **kwargs):
        encodes["n"] += 1
        return original(*args, **kwargs)

    policy.encode = wrapped  # type: ignore[method-assign]
    batched_logp, batched_value, _, _ = policy.evaluate_actions(states, actions)
    assert encodes["n"] == 1
    policy.encode = original  # type: ignore[method-assign]
    for index, action in enumerate(actions):
        assert batched_logp[index].item() == pytest.approx(
            float(policy.log_prob(state, action).item()), abs=1e-5
        )
        assert batched_value[index].item() == pytest.approx(
            float(policy.value(state).item()), abs=1e-5
        )


def test_cpu_cash_dirichlet_sample_is_seed_stable() -> None:
    torch.manual_seed(11)
    first = sample_cash_and_weights(
        k=2,
        selected_idx=(0, 1),
        order=("A", "B"),
        log_p_k=0.0,
        log_p_sel=0.0,
        cash_raw=torch.tensor([1.0, 1.5]),
        allocation_raw=torch.ones(4),
    )
    torch.manual_seed(11)
    second = sample_cash_and_weights(
        k=2,
        selected_idx=(0, 1),
        order=("A", "B"),
        log_p_k=0.0,
        log_p_sel=0.0,
        cash_raw=torch.tensor([1.0, 1.5]),
        allocation_raw=torch.ones(4),
    )
    assert first.z_cash == pytest.approx(second.z_cash, abs=1e-6)
    assert first.dirichlet_weights == pytest.approx(second.dirichlet_weights, abs=1e-6)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is not available"
)
def test_mps_k_positive_sample_and_log_prob_replay() -> None:
    assert "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ
    cash_raw = torch.tensor([1.0, 1.5], device="mps")
    allocation_raw = torch.ones(4, device="mps")
    action = sample_cash_and_weights(
        k=2,
        selected_idx=(0, 1),
        order=("A", "B"),
        log_p_k=0.0,
        log_p_sel=0.0,
        cash_raw=cash_raw,
        allocation_raw=allocation_raw,
    )
    assert action.k == 2
    mask = torch.tensor([True, True, False, False], device="mps")
    replayed = recompute_action_log_prob_tensors(
        action,
        count_logits=torch.zeros(16, device="mps"),
        selection_logits=torch.zeros(4, device="mps"),
        cash_raw=cash_raw,
        allocation_raw=allocation_raw,
        asset_mask=mask,
    )
    assert replayed.device.type == "mps"
    assert torch.isfinite(replayed)


def test_infer_decision_value_is_one_encode_and_matches_value() -> None:
    state = build_ppo_discovery_state(_request())
    policy = _tiny_policy()
    encodes = {"n": 0}
    original = policy.encode

    def wrapped(*args, **kwargs):
        encodes["n"] += 1
        return original(*args, **kwargs)

    policy.encode = wrapped  # type: ignore[method-assign]
    weights, order, fused_value = policy.infer_decision_value(state)
    assert encodes["n"] == 1
    policy.encode = original  # type: ignore[method-assign]
    assert fused_value == pytest.approx(float(policy.value(state).item()), abs=1e-5)
    weights2, order2 = policy.infer_decision(state)
    assert weights == weights2
    assert order == order2

"""Frozen temporal embedding cache tests."""

from __future__ import annotations

import pytest
import torch

from brain_api.core.ppo_discovery.config import PPODiscoveryConfig
from brain_api.core.ppo_discovery.policy import PPODiscoveryActorCritic
from brain_api.core.ppo_discovery.schemas import PPODiscoveryError
from brain_api.core.ppo_discovery.state_builder import build_ppo_discovery_state
from brain_api.core.ppo_discovery.temporal_cache import FrozenTemporalEmbeddingCache
from tests.core.ppo_discovery.test_state_builder import _request


def test_cache_reuses_as_of_and_splits_history_variants() -> None:
    policy = PPODiscoveryActorCritic(PPODiscoveryConfig(dropout=0.0))
    policy.freeze_temporal()
    cache = FrozenTemporalEmbeddingCache(policy)
    state = build_ppo_discovery_state(_request())
    first = cache.get(state, "normal")
    second = cache.get(state, "normal")
    assert first.data_ptr() == second.data_ptr()
    zero = cache.get(state, "zero_history")
    assert first.shape == zero.shape
    assert not torch.equal(first, zero)


def test_cache_rejects_trainable_encoder_and_clears_on_unfreeze() -> None:
    policy = PPODiscoveryActorCritic(PPODiscoveryConfig(dropout=0.0))
    with pytest.raises(PPODiscoveryError, match="frozen"):
        FrozenTemporalEmbeddingCache(policy)
    policy.freeze_temporal()
    cache = FrozenTemporalEmbeddingCache(policy)
    state = build_ppo_discovery_state(_request())
    cached = cache.get(state).clone()
    cache.clear()
    policy.unfreeze_temporal()
    with pytest.raises(PPODiscoveryError, match="unfreeze"):
        cache.get(state)
    policy.freeze_temporal()
    cache = FrozenTemporalEmbeddingCache(policy)
    fresh = cache.get(state)
    assert torch.allclose(cached, fresh, atol=1e-5)

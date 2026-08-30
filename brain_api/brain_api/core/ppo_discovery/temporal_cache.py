"""Device-resident frozen temporal encoder cache for ppo_discovery."""

from __future__ import annotations

from typing import Literal

import torch

from brain_api.core.ppo_discovery.policy import (
    PPODiscoveryActorCritic,
    tensors_from_state,
)
from brain_api.core.ppo_discovery.schemas import CanonicalPPOState, PPODiscoveryError

HistoryVariant = Literal["normal", "zero_history"]


class FrozenTemporalEmbeddingCache:
    """Cache ``[512, 64]`` temporal embeddings while the encoder is frozen."""

    def __init__(self, policy: PPODiscoveryActorCritic) -> None:
        if any(parameter.requires_grad for parameter in policy.temporal.parameters()):
            raise PPODiscoveryError(
                "FrozenTemporalEmbeddingCache requires a frozen temporal encoder"
            )
        self._policy = policy
        self._cache: dict[tuple[str, HistoryVariant], torch.Tensor] = {}

    def get(
        self,
        state: CanonicalPPOState,
        history_variant: HistoryVariant = "normal",
    ) -> torch.Tensor:
        if any(
            parameter.requires_grad for parameter in self._policy.temporal.parameters()
        ):
            raise PPODiscoveryError(
                "FrozenTemporalEmbeddingCache cannot be used after unfreeze"
            )
        key = (str(state.as_of), history_variant)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        device = next(self._policy.parameters()).device
        history, _features, _globals, _mask = tensors_from_state(state, device)
        if history_variant == "zero_history":
            history = torch.zeros_like(history)
        with torch.no_grad():
            embeddings = self._policy.temporal(history.unsqueeze(0))[0].detach()
        self._cache[key] = embeddings
        return embeddings

    def stack(
        self,
        states: list[CanonicalPPOState],
        history_variant: HistoryVariant = "normal",
    ) -> torch.Tensor:
        return torch.stack(
            [self.get(state, history_variant) for state in states], dim=0
        )

    def clear(self) -> None:
        self._cache.clear()


__all__ = ["FrozenTemporalEmbeddingCache"]

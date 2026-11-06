"""Conditional allocation policies using the actual PPO head and set encoder."""

from dataclasses import dataclass

import numpy as np
import torch

from brain_api.core.ppo_discovery.distributions import (
    _dirichlet_log_prob,
    _sample_cpu_if_mps,
    clamp_concentration,
)
from brain_api.core.ppo_discovery.policy import PPODiscoveryActorCritic
from brain_api.core.ppo_discovery.schemas import SampledAction

N = 10
CASH = 0.10
SYMBOLS = tuple(f"S{i:02d}" for i in range(N))


@dataclass
class AllocationContext:
    signal_slot: int
    visible: bool = True


class ConditionalAllocationPolicy(PPODiscoveryActorCritic):
    """Fixed K/set/cash; conditional likelihood contains only the stock simplex."""

    def __init__(self, config, representation="set"):
        super().__init__(config)
        self.representation = representation
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.allocation_head.parameters():
            parameter.requires_grad = True
        for parameter in self.value_head.parameters():
            parameter.requires_grad = True
        if representation == "set":
            for parameter in self.set_encoder.parameters():
                parameter.requires_grad = True

    def context_encoding(self, contexts):
        device = next(self.parameters()).device
        batch = len(contexts)
        signals = torch.zeros(batch, N, device=device)
        for i, context in enumerate(contexts):
            if context.visible:
                signals[i, context.signal_slot] = 1.0
        globals_ = torch.zeros(batch, 7, device=device)
        globals_[:, 2] = CASH
        if self.representation == "head":
            direction = torch.tensor([1.0, -1.0] * 64, device=device)
            encoded = signals.unsqueeze(-1) * direction
            pooled = torch.cat([encoded.mean(dim=1), globals_], dim=-1)
        else:
            tokens = torch.zeros(batch, N, 73, device=device)
            # A visible asset feature already supported by the production set encoder.
            tokens[:, :, 64] = signals
            mask = torch.ones(batch, N, dtype=torch.bool, device=device)
            encoded, pooled, _ = self.set_encoder(tokens, mask, globals_)
        selected_context = encoded.mean(dim=1, keepdim=True).expand(-1, N, -1)
        raw = self.allocation_head(
            torch.cat([encoded, selected_context], dim=-1)
        ).squeeze(-1)
        return clamp_concentration(raw), self.value_head(pooled).squeeze(-1)

    def sample_batch(self, contexts):
        self.eval()
        with torch.no_grad():
            alpha, values = self.context_encoding(contexts)
            simplex = _sample_cpu_if_mps(torch.distributions.Dirichlet(alpha)).to(
                alpha.device
            )
            logs = _dirichlet_log_prob(alpha, simplex)
        actions = []
        for i in range(len(contexts)):
            q = simplex[i].cpu().numpy()
            weights = {s: float((1 - CASH) * q[j]) for j, s in enumerate(SYMBOLS)}
            weights["CASH"] = CASH
            actions.append(
                SampledAction(
                    k=N,
                    selection_order=SYMBOLS,
                    selection_indices=tuple(range(N)),
                    z_cash=None,
                    dirichlet_weights=tuple(float(x) for x in q),
                    percentage_weights=weights,
                    log_p_k=0.0,
                    log_p_selection=0.0,
                    log_p_cash=0.0,
                    log_p_dirichlet=float(logs[i]),
                    log_p_total=float(logs[i]),
                )
            )
        return actions, values.cpu().numpy(), logs.cpu().numpy()

    def evaluate_actions(self, states, actions, *, temporal_embeddings=None):
        del temporal_embeddings
        alpha, values = self.context_encoding(states)
        simplex = torch.as_tensor(
            [a.dirichlet_weights for a in actions],
            device=alpha.device,
            dtype=alpha.dtype,
        )
        logs = _dirichlet_log_prob(alpha, simplex)
        zeros = torch.zeros_like(logs)
        return logs, values, zeros, zeros

    def deterministic_batch(self, contexts):
        self.eval()
        with torch.no_grad():
            alpha, _ = self.context_encoding(contexts)
            return (
                alpha / alpha.sum(dim=-1, keepdim=True)
            ).cpu().numpy(), alpha.cpu().numpy()


def context_returns(contexts, rng, noise=0.01):
    returns = np.full((len(contexts), N), -0.005)
    for i, context in enumerate(contexts):
        returns[i, context.signal_slot] = 0.04
    return returns + rng.normal(0, noise, size=returns.shape)

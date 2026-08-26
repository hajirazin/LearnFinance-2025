"""On-policy rollouts and GAE for ppo_discovery."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from brain_api.core.ppo_discovery.config import PPODiscoveryConfig
from brain_api.core.ppo_discovery.policy import PPODiscoveryActorCritic
from brain_api.core.ppo_discovery.schemas import CanonicalPPOState, SampledAction


@dataclass
class RolloutStep:
    """One weekly transition stored for a PPO update."""

    state: CanonicalPPOState
    action: SampledAction
    reward: float
    value: float
    log_p: float
    done: bool
    realized_net_return: float = 0.0


def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    *,
    gamma: float,
    gae_lambda: float,
    last_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return advantages and value targets. Population-std normalize later."""
    advantages = np.zeros(len(rewards), dtype=np.float64)
    gae = 0.0
    next_value = last_value
    for index in reversed(range(len(rewards))):
        mask = 0.0 if dones[index] else 1.0
        delta = rewards[index] + gamma * next_value * mask - values[index]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[index] = gae
        next_value = values[index]
    returns = advantages + np.asarray(values, dtype=np.float64)
    return advantages, returns


def normalize_advantages(advantages: np.ndarray) -> np.ndarray:
    std = float(advantages.std(ddof=0))
    mean = float(advantages.mean())
    return (advantages - mean) / (std + 1e-8)


def collect_rollout(
    policy: PPODiscoveryActorCritic,
    states: list[CanonicalPPOState],
    rewards: list[float],
    dones: list[bool],
    *,
    config: PPODiscoveryConfig,
) -> list[RolloutStep]:
    """Sample actions on a provided weekly trajectory (research env)."""
    del config
    steps: list[RolloutStep] = []
    policy.eval()
    for state, reward, done in zip(states, rewards, dones, strict=True):
        with torch.no_grad():
            action = policy.sample_action(state)
            value = float(policy.value(state).item())
            log_p = float(policy.log_prob(state, action).item())
        if not np.isfinite(log_p) or not np.isfinite(value):
            raise ValueError("non-finite value or log-probability during rollout")
        steps.append(
            RolloutStep(
                state=state,
                action=action,
                reward=float(reward),
                value=value,
                log_p=log_p,
                done=bool(done),
            )
        )
    return steps


__all__ = [
    "RolloutStep",
    "collect_rollout",
    "compute_gae",
    "normalize_advantages",
]

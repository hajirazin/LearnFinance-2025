"""Stage B PPO trainer for ppo_discovery."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW

from brain_api.core.ppo_discovery.config import PPODiscoveryConfig
from brain_api.core.ppo_discovery.policy import PPODiscoveryActorCritic
from brain_api.core.ppo_discovery.rollout import (
    RolloutStep,
    compute_gae,
    normalize_advantages,
)
from brain_api.core.ppo_discovery.schemas import PPODiscoveryError


def _assert_finite(tensor: torch.Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all():
        raise PPODiscoveryError(f"non-finite {name} during PPO update")


def ppo_update(
    policy: PPODiscoveryActorCritic,
    steps: list[RolloutStep],
    optimizer: AdamW,
    *,
    config: PPODiscoveryConfig,
    update_index: int,
) -> dict[str, float]:
    """One PPO optimization pass over a rollout."""
    if update_index >= config.freeze_encoder_updates:
        policy.unfreeze_temporal()
        for group in optimizer.param_groups:
            if group.get("name") == "encoder":
                group["lr"] = config.encoder_finetune_lr
    else:
        policy.freeze_temporal()

    rewards = [step.reward for step in steps]
    values = [step.value for step in steps]
    dones = [step.done for step in steps]
    advantages, returns = compute_gae(
        rewards, values, dones, gamma=config.gamma, gae_lambda=config.gae_lambda
    )
    advantages = normalize_advantages(advantages)
    indices = np.arange(len(steps))
    last_loss = 0.0
    policy.train()
    for _ in range(config.ppo_epochs):
        np.random.shuffle(indices)
        for start in range(0, len(steps), config.minibatch_size):
            batch = indices[start : start + config.minibatch_size]
            policy_losses = []
            value_losses = []
            for offset in batch:
                step = steps[int(offset)]
                new_logp = policy.log_prob(step.state, step.action)
                _assert_finite(new_logp, "log_probability")
                if abs(float(new_logp.item()) - step.log_p) > 50:
                    raise PPODiscoveryError(
                        "action log-probability reconstruction diverged"
                    )
                ratio = torch.exp(new_logp - step.log_p)
                adv = torch.tensor(advantages[int(offset)], dtype=new_logp.dtype)
                clipped = torch.clamp(
                    ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon
                )
                policy_losses.append(-torch.min(ratio * adv, clipped * adv))
                new_value = policy.value(step.state)
                target = torch.tensor(returns[int(offset)], dtype=new_value.dtype)
                value_losses.append((new_value - target).pow(2))
            loss = (
                torch.stack(policy_losses).mean()
                + config.value_loss_coef * torch.stack(value_losses).mean()
            )
            _assert_finite(loss, "ppo_loss")
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            optimizer.step()
            last_loss = float(loss.item())
    return {"ppo_loss": last_loss, "update_index": float(update_index)}


def train_ppo_discovery(
    policy: PPODiscoveryActorCritic,
    episode_fn: Callable[[PPODiscoveryActorCritic], list[RolloutStep]],
    *,
    config: PPODiscoveryConfig,
    seed: int,
) -> dict[str, float]:
    """Run PPO for ``config.total_timesteps`` closed-loop environment steps."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    encoder_params = list(policy.temporal.parameters())
    other_params = [
        parameter
        for name, parameter in policy.named_parameters()
        if not name.startswith("temporal.")
    ]
    optimizer = AdamW(
        [
            {"params": other_params, "lr": config.actor_lr, "name": "actor"},
            {"params": encoder_params, "lr": 0.0, "name": "encoder"},
        ],
        weight_decay=config.weight_decay,
    )
    policy.freeze_temporal()
    steps_done = 0
    update_index = 0
    last_metrics: dict[str, float] = {}
    while steps_done < config.total_timesteps:
        rollout = episode_fn(policy)
        if not rollout:
            raise PPODiscoveryError("PPO episode produced no transitions")
        last_metrics = ppo_update(
            policy, rollout, optimizer, config=config, update_index=update_index
        )
        steps_done += len(rollout)
        update_index += 1
    last_metrics["timesteps"] = float(steps_done)
    last_metrics["seed"] = float(seed)
    return last_metrics


__all__ = ["ppo_update", "train_ppo_discovery"]

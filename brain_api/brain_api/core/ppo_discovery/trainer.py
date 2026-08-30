"""Stage B PPO trainer for ppo_discovery."""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW

from brain_api.core.ppo_discovery.checkpoints import (
    move_optimizer_state_to_device,
    restore_rng_state,
)
from brain_api.core.ppo_discovery.config import PPODiscoveryConfig
from brain_api.core.ppo_discovery.policy import PPODiscoveryActorCritic
from brain_api.core.ppo_discovery.rollout import (
    RolloutStep,
    compute_gae,
    normalize_advantages,
)
from brain_api.core.ppo_discovery.schemas import PPODiscoveryError
from brain_api.core.ppo_discovery.temporal_cache import FrozenTemporalEmbeddingCache

logger = logging.getLogger(__name__)


def _assert_finite(tensor: torch.Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all():
        raise PPODiscoveryError(f"non-finite {name} during PPO update")


def _assert_gradient_devices(
    policy: PPODiscoveryActorCritic,
    device: torch.device,
) -> None:
    """Require every populated gradient to match its parameter and train device."""
    for name, parameter in policy.named_parameters():
        if parameter.grad is None:
            continue
        if parameter.device != device or parameter.grad.device != parameter.device:
            raise PPODiscoveryError(
                "gradient device mismatch for "
                f"{name}: param={parameter.device} grad={parameter.grad.device} "
                f"train={device}"
            )


def _device_of(policy: PPODiscoveryActorCritic) -> torch.device:
    return next(policy.parameters()).device


def ppo_update(
    policy: PPODiscoveryActorCritic,
    steps: list[RolloutStep],
    optimizer: AdamW,
    *,
    config: PPODiscoveryConfig,
    update_index: int,
    last_value: float = 0.0,
    cache: FrozenTemporalEmbeddingCache | None = None,
) -> dict[str, float]:
    """One PPO optimization pass over a rollout."""
    if update_index >= config.freeze_encoder_updates:
        if cache is not None:
            cache.clear()
            cache = None
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
        rewards,
        values,
        dones,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        last_value=last_value,
    )
    advantages = normalize_advantages(advantages)
    indices = np.arange(len(steps))
    device = _device_of(policy)
    policy.eval()
    last_loss = 0.0
    for _ in range(config.ppo_epochs):
        np.random.shuffle(indices)
        for start in range(0, len(steps), config.minibatch_size):
            batch_idx = indices[start : start + config.minibatch_size]
            semantic_len = len(batch_idx)
            batch_adv = torch.as_tensor(
                advantages[batch_idx], dtype=torch.float32, device=device
            )
            batch_ret = torch.as_tensor(
                returns[batch_idx], dtype=torch.float32, device=device
            )
            batch_old = torch.tensor(
                [steps[int(offset)].log_p for offset in batch_idx],
                dtype=torch.float32,
                device=device,
            )
            optimizer.zero_grad()
            for micro_start in range(0, semantic_len, config.ppo_microbatch_size):
                micro_end = min(micro_start + config.ppo_microbatch_size, semantic_len)
                micro_idx = batch_idx[micro_start:micro_end]
                micro_steps = [steps[int(offset)] for offset in micro_idx]
                embeddings = None
                if cache is not None:
                    embeddings = cache.stack([step.state for step in micro_steps])
                new_logp, new_value, h_count, h_sel = policy.evaluate_actions(
                    [step.state for step in micro_steps],
                    [step.action for step in micro_steps],
                    temporal_embeddings=embeddings,
                )
                _assert_finite(new_logp, "log_probability")
                old = batch_old[micro_start:micro_end]
                if ((new_logp - old).abs() > 50).any():
                    raise PPODiscoveryError(
                        "action log-probability reconstruction diverged"
                    )
                adv = batch_adv[micro_start:micro_end]
                ratio = torch.exp(new_logp - old)
                clipped = torch.clamp(
                    ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon
                )
                policy_term = -torch.min(ratio * adv, clipped * adv)
                policy_loss = policy_term.sum() / semantic_len
                target = batch_ret[micro_start:micro_end]
                value_loss = (new_value - target).pow(2).sum() / semantic_len
                _assert_finite(h_count, "count_entropy")
                _assert_finite(h_sel, "selection_entropy")
                loss = (
                    policy_loss
                    + config.value_loss_coef * value_loss
                    - config.count_entropy_coef * h_count.sum() / semantic_len
                    - config.selection_entropy_coef * h_sel.sum() / semantic_len
                )
                _assert_finite(loss, "ppo_loss")
                loss.backward()
                last_loss = float(loss.detach().item())
            _assert_gradient_devices(policy, device)
            nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            optimizer.step()
    return {"ppo_loss": last_loss, "update_index": float(update_index)}


def train_ppo_discovery(
    policy: PPODiscoveryActorCritic,
    episode_fn: Callable[..., list[RolloutStep]],
    *,
    config: PPODiscoveryConfig,
    seed: int,
    device: torch.device | None = None,
    resume: dict | None = None,
    on_episode_complete: Callable[..., None] | None = None,
    progress: Callable[[dict], None] | None = None,
) -> dict[str, float]:
    """Run PPO for ``config.total_timesteps`` closed-loop environment steps."""
    report = progress or (lambda _payload: None)
    device = device or _device_of(policy)
    policy.to(device)
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
    episode_index = 0
    last_metrics: dict[str, float] = {}
    if resume is not None:
        policy.load_state_dict(resume["policy"])
        policy.to(device)
        optimizer.load_state_dict(resume["optimizer"])
        move_optimizer_state_to_device(optimizer, device)
        steps_done = int(resume["steps_done"])
        update_index = int(resume["update_index"])
        episode_index = int(resume.get("episode_index", 0))
        restore_rng_state(resume["rng"], device)
        if update_index >= config.freeze_encoder_updates:
            policy.unfreeze_temporal()
            for group in optimizer.param_groups:
                if group.get("name") == "encoder":
                    group["lr"] = config.encoder_finetune_lr
        else:
            policy.freeze_temporal()
    else:
        torch.manual_seed(seed)
        np.random.seed(seed)
    cache = (
        FrozenTemporalEmbeddingCache(policy)
        if update_index < config.freeze_encoder_updates
        else None
    )
    log_interval = max(1, int(config.total_timesteps) // 10)
    next_log_at = ((steps_done // log_interval) + 1) * log_interval
    start_line = (
        f"[PPO] seed={seed} start steps_done={steps_done} "
        f"total_timesteps={config.total_timesteps} device={device.type}"
    )
    print(start_line, flush=True)
    logger.info(start_line)
    while steps_done < config.total_timesteps:
        rollout = episode_fn(policy, cache)
        if not rollout:
            raise PPODiscoveryError("PPO episode produced no transitions")
        horizon = config.rollout_length
        if horizon < 1:
            raise PPODiscoveryError("rollout_length must be at least 1")
        ep_return = float(sum(step.reward for step in rollout))
        ep_net = float(sum(step.realized_net_return for step in rollout))
        mean_k = float(np.mean([step.action.k for step in rollout]))
        for start in range(0, len(rollout), horizon):
            chunk = rollout[start : start + horizon]
            next_index = start + len(chunk)
            if next_index < len(rollout) and not chunk[-1].done:
                bootstrap = float(rollout[next_index].value)
            else:
                bootstrap = 0.0
            last_metrics = ppo_update(
                policy,
                chunk,
                optimizer,
                config=config,
                update_index=update_index,
                last_value=bootstrap,
                cache=cache,
            )
            update_index += 1
            if update_index >= config.freeze_encoder_updates and cache is not None:
                cache.clear()
                cache = None
        steps_done += len(rollout)
        episode_index += 1
        last_metrics["timesteps"] = float(steps_done)
        last_metrics["seed"] = float(seed)
        last_metrics["episode_index"] = float(episode_index)
        last_metrics["ep_return"] = ep_return
        last_metrics["ep_net"] = ep_net
        last_metrics["mean_k"] = mean_k
        if on_episode_complete is not None:
            on_episode_complete(
                policy=policy,
                optimizer=optimizer,
                steps_done=steps_done,
                episode_index=episode_index,
                update_index=update_index,
                metrics=last_metrics,
            )
        if steps_done >= next_log_at:
            encoder = (
                "unfrozen"
                if update_index >= config.freeze_encoder_updates
                else "frozen"
            )
            line = (
                f"[PPO] seed={seed} episode={episode_index} "
                f"steps={steps_done}/{config.total_timesteps} "
                f"ppo_loss={last_metrics.get('ppo_loss', 0.0):.4f} "
                f"ep_return={ep_return:.4f} ep_net={ep_net:.4f} "
                f"mean_k={mean_k:.2f} encoder={encoder} device={device.type}"
            )
            print(line, flush=True)
            logger.info(line)
            report(
                {
                    "stage": "ppo",
                    "seed": int(seed),
                    "steps_done": int(steps_done),
                    "total_timesteps": int(config.total_timesteps),
                    "episode_index": int(episode_index),
                    "device": device.type,
                }
            )
            while next_log_at <= steps_done:
                next_log_at += log_interval
    complete_line = (
        f"[PPO] seed={seed} complete steps={steps_done} "
        f"total_timesteps={config.total_timesteps} device={device.type}"
    )
    print(complete_line, flush=True)
    logger.info(complete_line)
    last_metrics["timesteps"] = float(steps_done)
    last_metrics["seed"] = float(seed)
    return last_metrics


__all__ = ["ppo_update", "train_ppo_discovery"]

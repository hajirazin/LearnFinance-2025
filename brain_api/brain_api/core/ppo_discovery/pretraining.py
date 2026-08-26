"""Stage A supervised pretraining of the temporal price encoder."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW

from brain_api.core.ppo_discovery.config import PPODiscoveryConfig
from brain_api.core.ppo_discovery.policy import PPODiscoveryActorCritic
from brain_api.core.ppo_discovery.schemas import PPODiscoveryError


def next_week_open_log_return(open_t: float, open_next: float) -> float:
    """Target: ``log(open[t+1] / open[t])``."""
    if (
        not np.isfinite(open_t)
        or not np.isfinite(open_next)
        or open_t <= 0
        or open_next <= 0
    ):
        raise PPODiscoveryError("pretrain target opens must be finite and positive")
    return float(np.log(open_next / open_t))


def rank_ic(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Cross-sectional Spearman rank IC (average-tie ranks via argsort)."""
    if len(predictions) < 2:
        return 0.0
    pred_rank = np.argsort(np.argsort(predictions))
    target_rank = np.argsort(np.argsort(targets))
    pred_rank = pred_rank.astype(np.float64)
    target_rank = target_rank.astype(np.float64)
    pred_rank -= pred_rank.mean()
    target_rank -= target_rank.mean()
    denom = np.sqrt((pred_rank**2).sum() * (target_rank**2).sum())
    if denom <= 0:
        return 0.0
    return float((pred_rank * target_rank).sum() / denom)


def pretrain_temporal_encoder(
    policy: PPODiscoveryActorCritic,
    histories: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    *,
    config: PPODiscoveryConfig,
    seed: int = 42,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Fit the temporal encoder with Smooth L1 on next-week open log returns.

    ``histories`` / ``targets`` are per-week arrays of shape
    ``[n_assets, 250, 4]`` and ``[n_assets]``. Assets with non-finite
    targets are ignored.
    """
    if not histories:
        raise PPODiscoveryError("pretraining requires at least one week")
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or next(policy.parameters()).device
    policy.train()
    optimizer = AdamW(
        list(policy.temporal.parameters()) + list(policy.pretrain_head.parameters()),
        lr=config.pretrain_lr,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.SmoothL1Loss(beta=config.pretrain_huber_beta)
    best_ic = -np.inf
    best_loss = np.inf
    best_state = {
        k: v.detach().cpu().clone() for k, v in policy.temporal.state_dict().items()
    }
    patience = 0
    n_weeks = len(histories)
    val_start = max(n_weeks - max(n_weeks // 5, 1), 0)

    def _epoch(indices: list[int], train: bool) -> tuple[float, float]:
        preds: list[float] = []
        targs: list[float] = []
        losses: list[float] = []
        for index in indices:
            history = torch.as_tensor(
                histories[index], dtype=torch.float32, device=device
            ).unsqueeze(0)
            target = torch.as_tensor(targets[index], dtype=torch.float32, device=device)
            valid = torch.isfinite(target)
            if not bool(valid.any()):
                continue
            pred = policy.pretrain_forward(history)[0]
            loss = loss_fn(pred[valid], target[valid])
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(policy.temporal.parameters())
                    + list(policy.pretrain_head.parameters()),
                    config.max_grad_norm,
                )
                optimizer.step()
            losses.append(float(loss.item()))
            preds.extend(pred[valid].detach().cpu().tolist())
            targs.extend(target[valid].detach().cpu().tolist())
        mean_loss = float(np.mean(losses)) if losses else float("inf")
        return mean_loss, rank_ic(np.asarray(preds), np.asarray(targs))

    for _epoch_i in range(config.pretrain_max_epochs):
        train_idx = list(range(val_start or n_weeks))
        val_idx = list(range(val_start, n_weeks)) if val_start else train_idx
        _epoch(train_idx, train=True)
        val_loss, val_ic = _epoch(val_idx, train=False)
        improved = val_ic > best_ic + 1e-8 or (
            abs(val_ic - best_ic) <= 1e-8 and val_loss < best_loss
        )
        if improved:
            best_ic = val_ic
            best_loss = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in policy.temporal.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= config.pretrain_patience:
                break
    policy.temporal.load_state_dict(best_state)
    return {"best_val_rank_ic": float(best_ic), "best_val_smooth_l1": float(best_loss)}


__all__ = ["next_week_open_log_return", "pretrain_temporal_encoder", "rank_ic"]

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
    """Cross-sectional Spearman rank IC using average ranks on ties."""
    if len(predictions) < 2:
        return 0.0
    pred_rank = _average_ranks(np.asarray(predictions, dtype=np.float64))
    target_rank = _average_ranks(np.asarray(targets, dtype=np.float64))
    pred_rank -= pred_rank.mean()
    target_rank -= target_rank.mean()
    denom = np.sqrt((pred_rank**2).sum() * (target_rank**2).sum())
    if denom <= 0:
        return 0.0
    return float((pred_rank * target_rank).sum() / denom)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    index = 0
    while index < len(values):
        end = index
        while end + 1 < len(values) and values[order[end + 1]] == values[order[index]]:
            end += 1
        average = (index + end) / 2.0 + 1.0
        ranks[order[index : end + 1]] = average
        index = end + 1
    return ranks


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
        week_ics: list[float] = []
        losses: list[float] = []
        batch_size = max(1, config.pretrain_batch_size)
        for index in indices:
            history = torch.as_tensor(
                histories[index], dtype=torch.float32, device=device
            ).unsqueeze(0)
            target = torch.as_tensor(targets[index], dtype=torch.float32, device=device)
            valid = torch.isfinite(target)
            if not bool(valid.any()):
                continue
            n_assets = int(history.size(1))
            week_preds = torch.empty(n_assets, device=device)
            for start in range(0, n_assets, batch_size):
                end = min(start + batch_size, n_assets)
                pred = policy.pretrain_forward(history[:, start:end])[0]
                week_preds[start:end] = pred
                slot_valid = valid[start:end]
                if not bool(slot_valid.any()):
                    continue
                loss = loss_fn(pred[slot_valid], target[start:end][slot_valid])
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
            finite = valid.detach().cpu().numpy()
            week_ics.append(
                rank_ic(
                    week_preds.detach().cpu().numpy()[finite],
                    target.detach().cpu().numpy()[finite],
                )
            )
        mean_loss = float(np.mean(losses)) if losses else float("inf")
        mean_ic = float(np.mean(week_ics)) if week_ics else 0.0
        return mean_loss, mean_ic

    for _epoch_i in range(config.pretrain_max_epochs):
        train_idx = list(range(val_start or n_weeks))
        val_idx = list(range(val_start, n_weeks)) if val_start else train_idx
        policy.train()
        _epoch(train_idx, train=True)
        policy.eval()
        with torch.no_grad():
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

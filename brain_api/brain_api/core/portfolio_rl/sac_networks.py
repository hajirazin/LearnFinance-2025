"""Masked, permutation-safe attention networks for SAC v3."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from brain_api.core.portfolio_rl.state import (
    ACTION_DIM,
    ASSET_FEATURES,
    GLOBAL_FEATURES,
    LEARNED_STATE_DIM,
    MAX_ASSETS,
    STATE_DIM,
)

LOG_STD_MIN = -20
LOG_STD_MAX = 2


def _activation(name: str) -> type[nn.Module]:
    activations = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}
    try:
        return activations[name]
    except KeyError as exc:
        raise ValueError(f"Unknown activation: {name}") from exc


def create_mlp(
    input_dim: int,
    output_dim: int,
    hidden_sizes: tuple[int, ...] = (64, 64),
    activation: str = "relu",
) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    activation_cls = _activation(activation)
    for size in hidden_sizes:
        layers.extend((nn.Linear(previous, size), activation_cls()))
        previous = size
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


def unpack_state_tensor(
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack a batch of serialized states without detaching gradients."""
    if state.ndim == 1:
        state = state.unsqueeze(0)
    if state.ndim != 2 or state.shape[-1] != STATE_DIM:
        raise ValueError(f"state must have trailing dimension {STATE_DIM}")
    assets_end = MAX_ASSETS * ASSET_FEATURES
    assets = state[:, :assets_end].reshape(-1, MAX_ASSETS, ASSET_FEATURES)
    globals_ = state[:, assets_end:LEARNED_STATE_DIM]
    raw_mask = state[:, LEARNED_STATE_DIM:]
    mask = raw_mask > 0.5
    if torch.any(mask.sum(dim=1) < 1):
        raise ValueError("each SAC state must contain at least one valid asset")
    return assets, globals_, mask


class MaskedStateEncoder(nn.Module):
    """Shared token encoder followed by masked cross-asset attention."""

    def __init__(self, hidden_dim: int, activation: str) -> None:
        super().__init__()
        activation_cls = _activation(activation)
        self.token_encoder = nn.Sequential(
            nn.Linear(ASSET_FEATURES, hidden_dim), activation_cls()
        )
        heads = 4 if hidden_dim % 4 == 0 else 1
        self.attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)

    def forward(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assets, globals_, mask = unpack_state_tensor(state)
        encoded = self.token_encoder(assets)
        attended, _ = self.attention(
            encoded,
            encoded,
            encoded,
            key_padding_mask=~mask,
            need_weights=False,
        )
        mask_f = mask.unsqueeze(-1).to(attended.dtype)
        attended = attended * mask_f
        pooled = attended.sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        return attended, pooled, globals_, mask


class GaussianActor(nn.Module):
    """Equivariant masked Gaussian policy over 30 stock slots plus CASH."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        hidden_sizes: tuple[int, ...] = (64, 64),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if state_dim != STATE_DIM or action_dim != ACTION_DIM:
            raise ValueError(
                f"SAC v3 requires state_dim={STATE_DIM}, action_dim={ACTION_DIM}"
            )
        hidden_dim = hidden_sizes[0] if hidden_sizes else 64
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.encoder = MaskedStateEncoder(hidden_dim, activation)
        context_dim = hidden_dim * 2 + GLOBAL_FEATURES
        cash_context_dim = hidden_dim + GLOBAL_FEATURES
        self.asset_mean = nn.Linear(context_dim, 1)
        self.asset_log_std = nn.Linear(context_dim, 1)
        self.cash_mean = nn.Linear(cash_context_dim, 1)
        self.cash_log_std = nn.Linear(cash_context_dim, 1)

    def distribution_parameters(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attended, pooled, globals_, mask = self.encoder(state)
        repeated = pooled.unsqueeze(1).expand(-1, MAX_ASSETS, -1)
        repeated_globals = globals_.unsqueeze(1).expand(-1, MAX_ASSETS, -1)
        context = torch.cat((attended, repeated, repeated_globals), dim=-1)
        asset_mean = self.asset_mean(context).squeeze(-1)
        asset_log_std = self.asset_log_std(context).squeeze(-1)
        cash_context = torch.cat((pooled, globals_), dim=-1)
        mean = torch.cat((asset_mean, self.cash_mean(cash_context)), dim=-1)
        log_std = torch.cat(
            (asset_log_std, self.cash_log_std(cash_context)), dim=-1
        ).clamp(LOG_STD_MIN, LOG_STD_MAX)
        canonical = torch.cat(
            (
                mask,
                torch.ones((mask.shape[0], 1), dtype=torch.bool, device=mask.device),
            ),
            dim=-1,
        )
        mean = mean.masked_fill(~canonical, 0.0)
        log_std = log_std.masked_fill(~canonical, 0.0)
        return mean, log_std, canonical

    def forward(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std, action_mask = self.distribution_parameters(state)
        if deterministic:
            return torch.tanh(mean).masked_fill(~action_mask, 0.0), torch.zeros(
                mean.shape[0], device=mean.device
            )
        distribution = Normal(mean, log_std.exp())
        pre_tanh = distribution.rsample()
        action = torch.tanh(pre_tanh).masked_fill(~action_mask, 0.0)
        per_dimension = distribution.log_prob(pre_tanh) - torch.log(
            1 - torch.tanh(pre_tanh).pow(2) + 1e-6
        )
        log_prob = (per_dimension * action_mask.to(per_dimension.dtype)).sum(dim=-1)
        return action, log_prob

    def get_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            device = next(self.parameters()).device
            tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
            single = tensor.ndim == 1
            action, _ = self.forward(tensor, deterministic=deterministic)
            result = action.cpu().numpy()
            return result[0] if single else result


class Critic(nn.Module):
    """Permutation-invariant Q network over shared masked token/action pairs."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        hidden_sizes: tuple[int, ...] = (64, 64),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if state_dim != STATE_DIM or action_dim != ACTION_DIM:
            raise ValueError(
                f"SAC v3 requires state_dim={STATE_DIM}, action_dim={ACTION_DIM}"
            )
        hidden_dim = hidden_sizes[0] if hidden_sizes else 64
        activation_cls = _activation(activation)
        self.token_encoder = nn.Sequential(
            nn.Linear(ASSET_FEATURES + 1, hidden_dim), activation_cls()
        )
        remainder = hidden_sizes[1:] if len(hidden_sizes) > 1 else (hidden_dim,)
        self.q_net = create_mlp(
            hidden_dim + GLOBAL_FEATURES + 1,
            1,
            tuple(remainder),
            activation,
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        assets, globals_, mask = unpack_state_tensor(state)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if action.shape != (assets.shape[0], ACTION_DIM):
            raise ValueError(f"action must have shape (batch, {ACTION_DIM})")
        masked_actions = action[:, :MAX_ASSETS].masked_fill(~mask, 0.0)
        pairs = torch.cat((assets, masked_actions.unsqueeze(-1)), dim=-1)
        encoded = self.token_encoder(pairs)
        mask_f = mask.unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * mask_f).sum(1) / mask_f.sum(1).clamp_min(1.0)
        return self.q_net(torch.cat((pooled, globals_, action[:, -1:]), dim=-1))


class TwinCritic(nn.Module):
    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        hidden_sizes: tuple[int, ...] = (64, 64),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.q1 = Critic(state_dim, action_dim, hidden_sizes, activation)
        self.q2 = Critic(state_dim, action_dim, hidden_sizes, activation)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(state, action), self.q2(state, action)

    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q1(state, action)

    def min_q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.forward(state, action))


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(
            target.parameters(), source.parameters(), strict=True
        ):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(source.state_dict())

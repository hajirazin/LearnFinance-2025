"""Shared temporal patch encoder for ppo_discovery."""

from __future__ import annotations

import math

import torch
from torch import nn

from brain_api.core.ppo_discovery.config import (
    ENCODER_CHANNELS,
    ENCODER_SESSIONS,
    N_PATCHES,
    PATCH_LENGTH,
    TEMPORAL_D_MODEL,
)


def sinusoidal_positions(length: int, width: int, device: torch.device) -> torch.Tensor:
    """Fixed sinusoidal temporal positions; required because order in time matters."""
    position = torch.arange(length, device=device).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, width, 2, device=device).float() * (-math.log(10000.0) / width)
    )
    pe = torch.zeros(length, width, device=device)
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
    return pe


class PPODiscoveryTemporalEncoder(nn.Module):
    """Patch 250x4 into 50 non-overlapping weekly patches, then mean-pool."""

    def __init__(
        self,
        *,
        d_model: int = TEMPORAL_D_MODEL,
        n_heads: int = 4,
        n_layers: int = 2,
        ffn_dim: int = 128,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.patch_length = PATCH_LENGTH
        self.n_patches = N_PATCHES
        self.projection = nn.Linear(PATCH_LENGTH * ENCODER_CHANNELS, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """history: [batch, assets, 250, 4] -> [batch, assets, 64]."""
        if history.ndim != 4:
            raise ValueError("history must be [batch, assets, 250, 4]")
        batch, assets, sessions, channels = history.shape
        if sessions != ENCODER_SESSIONS or channels != ENCODER_CHANNELS:
            raise ValueError("history trailing shape must be (250, 4)")
        patched = history.reshape(
            batch, assets, self.n_patches, self.patch_length * ENCODER_CHANNELS
        )
        tokens = self.projection(patched)
        positions = sinusoidal_positions(self.n_patches, tokens.size(-1), tokens.device)
        tokens = self.dropout(tokens + positions.view(1, 1, self.n_patches, -1))
        flat = tokens.reshape(batch * assets, self.n_patches, tokens.size(-1))
        encoded = self.encoder(flat)
        pooled = encoded.mean(dim=1)
        return pooled.reshape(batch, assets, -1)


__all__ = ["PPODiscoveryTemporalEncoder", "sinusoidal_positions"]

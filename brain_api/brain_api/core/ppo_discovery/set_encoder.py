"""Permutation-equivariant cross-asset encoder (no asset positional encoding)."""

from __future__ import annotations

import torch
from torch import nn

from brain_api.core.ppo_discovery.config import (
    GLOBAL_FEATURES,
    SET_D_MODEL,
    TOKEN_WIDTH,
)


class PPODiscoverySetEncoder(nn.Module):
    """Masked set Transformer over 64+9 asset tokens plus 7 globals."""

    def __init__(
        self,
        *,
        token_width: int = TOKEN_WIDTH,
        d_model: int = SET_D_MODEL,
        n_heads: int = 4,
        n_layers: int = 2,
        ffn_dim: int = 256,
        dropout: float = 0.10,
        n_globals: int = GLOBAL_FEATURES,
    ) -> None:
        super().__init__()
        self.token_proj = nn.Linear(token_width, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )
        self.n_globals = n_globals

    def forward(
        self, tokens: torch.Tensor, asset_mask: torch.Tensor, globals_: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return per-asset encodings, pooled state, and globals.

        tokens: [batch, assets, 73]
        asset_mask: [batch, assets] True = eligible
        globals_: [batch, 7]
        """
        if tokens.ndim != 3:
            raise ValueError("tokens must be [batch, assets, width]")
        encoded = self.encoder(
            self.token_proj(tokens),
            src_key_padding_mask=~asset_mask,
        )
        encoded = encoded * asset_mask.unsqueeze(-1)
        denom = asset_mask.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(-1)
        pooled_assets = encoded.sum(dim=1, keepdim=False) / denom.squeeze(-1)
        pooled = torch.cat([pooled_assets, globals_], dim=-1)
        return encoded, pooled, globals_


__all__ = ["PPODiscoverySetEncoder"]

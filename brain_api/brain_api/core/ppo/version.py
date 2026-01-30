"""PPO version computation.

Deterministic version string based on training parameters.
"""

from __future__ import annotations

import hashlib
from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brain_api.core.ppo.config import PPOConfig


def compute_version(
    start_date: date,
    end_date: date,
    symbols: list[str],
    config: PPOConfig,
) -> str:
    """Compute deterministic version string for PPO model.

    Version format: v{end_date}-{hash}

    The hash is computed from all inputs that affect the model,
    ensuring reproducibility and idempotency.

    Args:
        start_date: Training data start date.
        end_date: Training data end date.
        symbols: List of symbols used for training.
        config: PPO configuration.

    Returns:
        Version string like "v2025-01-08-abc123def456".
    """
    # Create a stable representation of inputs
    hash_input = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "symbols": sorted(symbols),  # Sort for stability
        "config": config.to_dict(),
        "model_type": "ppo",
    }

    # Create hash
    hash_str = str(hash_input).encode("utf-8")
    hash_digest = hashlib.sha256(hash_str).hexdigest()[:12]

    return f"v{end_date.isoformat()}-{hash_digest}"

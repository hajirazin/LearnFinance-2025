"""Version computation for SAC + LSTM models.

Creates deterministic version strings based on data window, symbols, and config.
"""

import hashlib
from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brain_api.core.sac_lstm.config import SACLSTMConfig


def compute_version(
    data_window_start: date,
    data_window_end: date,
    symbols: list[str],
    config: "SACLSTMConfig",
) -> str:
    """Compute deterministic version string for SAC + LSTM model.

    Version format: v{date}_{hash}
    where hash is derived from data window, symbols, and key config params.

    Args:
        data_window_start: Start date of training data.
        data_window_end: End date of training data.
        symbols: List of symbols used for training.
        config: SAC + LSTM configuration.

    Returns:
        Version string (e.g., "v2026-01-08_abc123")
    """
    # Build deterministic hash input
    hash_input = (
        f"sac_lstm_{data_window_start.isoformat()}_{data_window_end.isoformat()}_"
        f"{'_'.join(sorted(symbols))}_"
        f"hidden_{config.hidden_sizes}_"
        f"actor_lr_{config.actor_lr}_"
        f"critic_lr_{config.critic_lr}_"
        f"tau_{config.tau}_"
        f"gamma_{config.gamma}_"
        f"alpha_{config.init_alpha}_"
        f"seed_{config.seed}"
    )

    # Compute short hash
    hash_bytes = hashlib.sha256(hash_input.encode()).digest()
    short_hash = hash_bytes[:4].hex()

    # Version string
    version = f"v{data_window_end.isoformat()}_{short_hash}"

    return version


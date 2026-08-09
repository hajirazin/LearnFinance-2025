"""Version computation for SAC models.

Creates deterministic version strings based on data window, symbols, and config.
"""

import hashlib
import json
from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brain_api.core.sac.config import SACConfig


def compute_version(
    data_window_start: date,
    data_window_end: date,
    symbols: list[str],
    config: "SACConfig",
) -> str:
    """Compute deterministic version string for SAC model.

    Version format: v{date}_{hash}
    where hash is derived from data window, symbols, and key config params.

    Args:
        data_window_start: Start date of training data.
        data_window_end: End date of training data.
        symbols: List of symbols used for training.
        config: SAC configuration.

    Returns:
        Version string (e.g., "v2026-01-08_abc123")
    """
    hash_input = json.dumps(
        {
            "model": "sac",
            "sac_schema_version": 3,
            "architecture": "masked_attention",
            "max_assets": 30,
            "action_dim": 31,
            "hmm": {
                "states": 3,
                "covariance": "diag",
                "seed": 42,
                "max_iterations": 200,
                "tolerance": 1e-4,
            },
            "data_window_start": data_window_start.isoformat(),
            "data_window_end": data_window_end.isoformat(),
            "symbols": sorted(symbols),
            "config": config.to_dict(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )

    # Compute short hash
    hash_bytes = hashlib.sha256(hash_input.encode()).digest()
    short_hash = hash_bytes[:4].hex()

    # Version string
    version = f"v{data_window_end.isoformat()}_{short_hash}"

    return version

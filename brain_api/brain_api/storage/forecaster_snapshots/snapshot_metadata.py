"""Snapshot ``metadata.json`` factory (forecaster checkpoints)."""

from datetime import UTC, date, datetime
from typing import Any


def create_snapshot_metadata(
    forecaster_type: str,
    cutoff_date: date,
    data_window_start: str,
    data_window_end: str,
    symbols: list[str],
    config: Any,
    train_loss: float,
    val_loss: float,
    *,
    config_symbols_hash: str,
) -> dict[str, Any]:
    """Create metadata dictionary for a forecaster snapshot."""

    return {
        "forecaster_type": forecaster_type,
        "cutoff_date": cutoff_date.isoformat(),
        "training_timestamp": datetime.now(UTC).isoformat(),
        "data_window": {
            "start": data_window_start,
            "end": data_window_end,
        },
        "symbols": symbols,
        "config": config.to_dict(),
        "config_symbols_hash": config_symbols_hash,
        "metrics": {
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
    }

"""Shared version computation for all model types."""

import hashlib
import json
from datetime import date
from typing import Any


def compute_model_hash(
    model_type: str,
    start_date: date,
    end_date: date,
    symbols: list[str],
    config_dict: dict[str, Any],
) -> str:
    """Deterministic 12-char digest of ``(model_type, window, symbols, config)``.

    Used inside :func:`compute_model_version` for main training versions.

    Args:
        model_type: Model / bucket discriminator (``"lstm"``, ``"patchtst"``,
            ``"lstm_halal_new"``, etc.).
        start_date: Training / snapshot-window start date
        end_date: Training window end date or snapshot cutoff
        symbols: Ticker symbols
        config_dict: Model configuration dictionary

    Returns:
        Twelve hex chars (sha256 truncation).
    """
    canonical = {
        "model": model_type,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "symbols": sorted(symbols),
        "config": config_dict,
    }
    canonical_json = json.dumps(canonical, sort_keys=True)
    return hashlib.sha256(canonical_json.encode()).hexdigest()[:12]


def compute_snapshot_identity_hash(
    model_type: str,
    cutoff_date: date,
    config_dict: dict[str, Any],
) -> str:
    """Return the deterministic snapshot identity for bucket, cutoff, and config.

    Args:
        model_type: Canonical snapshot bucket name.
        cutoff_date: Point-in-time snapshot cutoff.
        config_dict: Forecaster configuration dictionary.

    Returns:
        Twelve lowercase hexadecimal characters from a truncated SHA-256 digest.
    """
    canonical = {
        "model": model_type,
        "cutoff_date": cutoff_date.isoformat(),
        "config": config_dict,
    }
    canonical_json = json.dumps(canonical, sort_keys=True)
    return hashlib.sha256(canonical_json.encode()).hexdigest()[:12]


def compute_model_version(
    model_type: str,
    start_date: date,
    end_date: date,
    symbols: list[str],
    config_dict: dict[str, Any],
) -> str:
    """Compute deterministic version string for any model type.

    The version is a hash of (model_type, window, symbols, config) so that reruns
    with the same inputs produce the same version (idempotent training).

    Args:
        model_type: Type of model ("lstm" or "patchtst")
        start_date: Training data start date
        end_date: Training data end date
        symbols: List of ticker symbols
        config_dict: Model configuration as dictionary

    Returns:
        Version string in format 'v{date_prefix}-{hash_suffix}'
    """
    hash_digest = compute_model_hash(
        model_type, start_date, end_date, symbols, config_dict
    )
    date_prefix = end_date.strftime("%Y-%m-%d")
    return f"v{date_prefix}-{hash_digest}"

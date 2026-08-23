"""Shared metadata utilities for model storage.

This module provides a unified metadata factory for all model types.
"""

from datetime import UTC, date, datetime
from typing import Any

from brain_api.core.version import compute_model_hash


def create_training_metadata(
    model_type: str,
    version: str,
    data_window_start: str,
    data_window_end: str,
    symbols: list[str],
    config_dict: dict[str, Any],
    train_loss: float,
    val_loss: float,
    baseline_loss: float,
    best_epoch: int,
    stopped_epoch: int,
    promoted: bool,
    prior_version: str | None,
    failure_reasons: list[str] | None = None,
    config_symbols_hash: str | None = None,
    val_rank_ic: float | None = None,
) -> dict[str, Any]:
    """Create metadata dict for a training run.

    This is a unified factory that works for all model types (LSTM, PatchTST, etc.).

    Args:
        model_type: Registry bucket name ``{model}_{universe}`` (e.g.
            ``lstm_halal_new``, ``patchtst_halal_new``,
            ``patchtst_nifty_shariah_500``); same argument as
            :func:`~brain_api.core.version.compute_model_hash` receives as
            ``model_type``.
        version: Version string
        data_window_start: Training data start date (ISO format)
        data_window_end: Training data end date (ISO format)
        symbols: List of symbols used for training
        config_dict: Model configuration as dictionary
        train_loss: Final training loss
        val_loss: Validation loss
        baseline_loss: Baseline (persistence) loss
        best_epoch: 1-indexed epoch of the restored checkpoint (0 if none)
        stopped_epoch: 1-indexed last epoch actually run (0 if none)
        promoted: Whether this version was promoted to current
        prior_version: Previous current version (if any)
        failure_reasons: Human-readable strings explaining why
            ``promoted`` is ``False`` (empty when promoted is True).
            Defaults to ``[]`` so existing callers that have not
            migrated to the always-promote-with-guardrails policy
            still work.
        config_symbols_hash: Twelve-char digest of
            ``(model_type_bucket, window, symbols, config)`` for audit;
            mirrors forecaster snapshot folder suffixes. When omitted,
            computed automatically from the other fields.
        val_rank_ic: PatchTST validation weekly rank IC of the restored
            checkpoint. Omitted from metrics when None so LSTM hashes and
            callers stay unchanged.

    Returns:
        Metadata dictionary
    """
    ws = date.fromisoformat(data_window_start)
    we = date.fromisoformat(data_window_end)
    audit_hash = config_symbols_hash or compute_model_hash(
        model_type, ws, we, symbols, config_dict
    )
    metrics: dict[str, Any] = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "baseline_loss": baseline_loss,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
    }
    if val_rank_ic is not None:
        metrics["val_rank_ic"] = val_rank_ic
    return {
        "model_type": model_type,
        "version": version,
        "training_timestamp": datetime.now(UTC).isoformat(),
        "data_window": {
            "start": data_window_start,
            "end": data_window_end,
        },
        "symbols": symbols,
        "config": config_dict,
        "config_symbols_hash": audit_hash,
        "metrics": metrics,
        "promoted": promoted,
        "prior_version": prior_version,
        "failure_reasons": list(failure_reasons) if failure_reasons else [],
    }

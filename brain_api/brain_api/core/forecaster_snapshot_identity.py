"""Stable snapshot-hash inputs for SAC US walk-forward and forecaster trains.

Monthly forecaster snapshots for year-end cutoffs use the extended backfill
price window start (matching ``_backfill_lstm_snapshots`` /
``_backfill_patchtst_snapshots``). SAC dual-forecast generation only consumes
those Dec-31 checkpoints from the ``lstm_halal_new`` and ``patchtst_halal_new``
buckets via :func:`~brain_api.storage.policy.ensure_snapshot_for_bucket`.
"""

from __future__ import annotations

from datetime import date

from brain_api.core.config import resolve_training_window
from brain_api.core.lstm.config import DEFAULT_CONFIG as LSTM_DEFAULT_CONFIG
from brain_api.core.model_buckets import ModelType, get_bucket
from brain_api.core.patchtst.config import DEFAULT_CONFIG as PATCHTST_DEFAULT_CONFIG
from brain_api.core.version import compute_model_hash

# Must stay aligned with ``bootstrap_years`` in the LSTM/PatchTST backfill loops.
_SNAPSHOT_BACKFILL_BOOTSTRAP_YEARS = 4


def extended_backfill_window_start_date() -> date:
    """First calendar day loaded for RL snapshot backfill (extended window).

    Mirrors ``routes/training/lstm.py::_backfill_lstm_snapshots`` and
    ``routes/training/patchtst.py::_backfill_patchtst_snapshots``.
    """
    start_date, _ = resolve_training_window()
    start_year = start_date.year
    first_snapshot_year = start_year - 1
    return date(
        first_snapshot_year - _SNAPSHOT_BACKFILL_BOOTSTRAP_YEARS,
        1,
        1,
    )


def halal_new_lstm_resolver_symbols() -> list[str]:
    return list(get_bucket(ModelType.LSTM, "halal_new").symbols_resolver())


def halal_new_patchtst_resolver_symbols() -> list[str]:
    return list(get_bucket(ModelType.PATCHTST, "halal_new").symbols_resolver())


def expected_dec31_walkforward_snapshot_hash(
    *,
    forecaster_bucket: str,
    cutoff_date: date,
    resolver_symbols: list[str],
    config_dict: dict,
) -> str:
    """12-char digest for ``snapshot-{cutoff}-{digest}/`` (Dec-31 backfill rows)."""
    window_start = extended_backfill_window_start_date()
    return compute_model_hash(
        forecaster_bucket,
        window_start,
        cutoff_date,
        resolver_symbols,
        config_dict,
    )


def lstm_walkforward_expectation_bundle() -> tuple[str, list[str], dict]:
    """(bucket_name, resolver_symbols, default_lstm_config_dict) for SAC LSTM snaps."""
    return (
        "lstm_halal_new",
        halal_new_lstm_resolver_symbols(),
        LSTM_DEFAULT_CONFIG.to_dict(),
    )


def patchtst_walkforward_expectation_bundle() -> tuple[str, list[str], dict]:
    """(bucket_name, resolver_symbols, default_patchtst_config_dict)."""
    return (
        "patchtst_halal_new",
        halal_new_patchtst_resolver_symbols(),
        PATCHTST_DEFAULT_CONFIG.to_dict(),
    )

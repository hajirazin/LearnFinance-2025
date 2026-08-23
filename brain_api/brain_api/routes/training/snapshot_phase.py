"""Forecaster snapshot phase shared by LSTM + PatchTST training routes.

Houses the snapshot-phase helpers that the main-training background
runners and the snapshots-only background runners both depend on:

* Per-family ``_*MainTrainingArtifacts`` dataclasses bundle the in-memory
  outputs of a successful main training pass that the end-of-window
  snapshot writer consumes.
* Per-family ``_run_*_snapshot_phase`` orchestrates "write end-window
  (or warn-and-skip) -> backfill historical Dec-31 cutoffs". Both
  branches share the ``StoragePolicy``-aware existence-check rule via
  :func:`brain_api.core.forecaster_snapshot_identity._resolve_check_hf`.
* Per-family ``_backfill_*_snapshots`` re-trains the missing year-end
  snapshots only. Both functions accept ``policy: StoragePolicy | None
  = None`` so callers can override the env-var default for tests.

Splitting these out of the route files keeps both ``routes/training/lstm.py``
and ``routes/training/patchtst.py`` under the AGENTS.md 600-line ceiling.
The runners (``_run_*_snapshots_only``) and route handlers stay in the
route files because they perform FastAPI-flavored orchestration
(``update_progress`` / ``complete_job``); only the pure snapshot mechanics
live here.
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd
import torch

from brain_api.core.forecaster_snapshot_identity import _resolve_check_hf
from brain_api.core.lstm import (
    LSTMConfig,
    build_dataset,
    load_prices_yfinance,
    train_model_pytorch,
)
from brain_api.core.patchtst import (
    PatchTSTConfig,
    align_multivariate_data,
)
from brain_api.core.patchtst import (
    build_dataset as patchtst_build_dataset,
)
from brain_api.core.patchtst import (
    load_prices_yfinance as patchtst_load_prices,
)
from brain_api.core.patchtst import (
    train_model_pytorch as patchtst_train_model,
)
from brain_api.core.version import compute_model_hash
from brain_api.routes.training.snapshot_persist import persist_forecaster_snapshot
from brain_api.storage.forecaster_snapshots import (
    SnapshotLocalStorage,
    create_snapshot_metadata,
)
from brain_api.storage.policy import (
    StoragePolicy,
    get_storage_policy,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def _filter_prices_by_cutoff(
    prices: dict[str, pd.DataFrame],
    cutoff_date: date,
) -> dict[str, pd.DataFrame]:
    """Filter price DataFrames to include only data up to ``cutoff_date``.

    Used by both LSTM and PatchTST backfill loops -- the math is
    identical for both (filter by the DatetimeIndex), so keeping one
    copy is safe per AGENTS.md rule #2 (provably-identical filter).

    yfinance returns a tz-aware DatetimeIndex (``America/New_York``)
    while the backfill cutoff is a naive ``date``. Comparing them
    directly raises pandas' ``Invalid comparison between
    dtype=datetime64[ns, America/New_York] and Timestamp``. Localize
    the cutoff to each symbol's index tz before comparing -- mirrors
    the canonical pattern at
    :mod:`brain_api.core.lstm.inference` lines 89-91.

    Symbols that have no rows after filtering are dropped.
    """
    cutoff_ts = pd.Timestamp(cutoff_date)
    out: dict[str, pd.DataFrame] = {}
    for symbol, df in prices.items():
        symbol_cutoff = cutoff_ts
        if df.index.tz is not None and symbol_cutoff.tz is None:
            symbol_cutoff = symbol_cutoff.tz_localize(df.index.tz)
        filtered = df[df.index <= symbol_cutoff]
        if len(filtered) > 0:
            out[symbol] = filtered.copy()
    return out


def _filter_signals_by_cutoff(
    signals: dict[str, pd.DataFrame],
    cutoff_date: date,
) -> dict[str, pd.DataFrame]:
    """Filter signal DataFrames to include only data up to ``cutoff_date``.

    PatchTST-only helper. Works with both ``DatetimeIndex`` and regular
    indexes (will try to convert).

    Symbols with no rows after filtering are dropped. Symbols whose
    index cannot be parsed fall back to the raw DataFrame.
    """
    cutoff_ts = pd.Timestamp(cutoff_date)
    result = {}

    for symbol, df in signals.items():
        if df.empty:
            continue

        if isinstance(df.index, pd.DatetimeIndex):
            symbol_cutoff = cutoff_ts
            if df.index.tz is not None and symbol_cutoff.tz is None:
                symbol_cutoff = symbol_cutoff.tz_localize(df.index.tz)
            filtered = df[df.index <= symbol_cutoff]
        else:
            try:
                idx = pd.to_datetime(df.index)
                symbol_cutoff = cutoff_ts
                if (
                    isinstance(idx, pd.DatetimeIndex)
                    and idx.tz is not None
                    and symbol_cutoff.tz is None
                ):
                    symbol_cutoff = symbol_cutoff.tz_localize(idx.tz)
                mask = idx <= symbol_cutoff
                filtered = df[mask]
            except (ValueError, TypeError):
                filtered = df

        if len(filtered) > 0:
            result[symbol] = filtered.copy()

    return result


# ---------------------------------------------------------------------------
# LSTM snapshot phase
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LSTMMainTrainingArtifacts:
    """In-memory LSTM main-training outputs piggybacked into the
    end-of-window snapshot. ``None`` in the snapshots-only path."""

    model: Any
    feature_scaler: Any
    train_loss: float
    val_loss: float
    best_epoch: int
    stopped_epoch: int
    available_symbols: list[str]


def _run_lstm_snapshot_phase(
    *,
    train_window: tuple[date, date],
    symbols: list[str],
    config: LSTMConfig,
    snapshot_storage: SnapshotLocalStorage,
    main_artifacts: _LSTMMainTrainingArtifacts | None,
    policy: StoragePolicy | None = None,
    log_prefix: str = "[LSTM]",
) -> None:
    """Snapshot phase shared by main training and snapshots-only reruns.

    * ``main_artifacts`` set: write end-window snapshot with the
      in-memory model (byte-equivalent to the legacy in-line block).
    * ``main_artifacts`` ``None`` (snapshots-only path): warn-and-skip
      the end-window snapshot if missing, because regenerating it
      would require retraining main. The operator can delete the
      cached main version and rerun ``/train/lstm`` to recreate it.

    Historical backfill (``_backfill_lstm_snapshots``) always runs.

    ``policy`` (default ``get_storage_policy()``) is threaded through
    both the end-window check and the backfill loop. Under
    ``hf_first`` with no HF repo configured ``_resolve_check_hf``
    raises ``StoragePolicyError``; the caller routes that to
    ``fail_job``.
    """
    if policy is None:
        policy = get_storage_policy()
    start_date, end_date = train_window
    snapshot_hf_repo = snapshot_storage._get_hf_repo()
    check_hf = _resolve_check_hf(snapshot_storage=snapshot_storage, policy=policy)

    end_snap_digest = compute_model_hash(
        snapshot_storage.forecaster_type,
        start_date,
        end_date,
        symbols,
        config.to_dict(),
    )

    end_window_present = snapshot_storage.snapshot_exists_anywhere(
        end_date,
        end_snap_digest,
        check_hf=check_hf,
    )

    if not end_window_present:
        if main_artifacts is not None:
            snapshot_metadata = create_snapshot_metadata(
                forecaster_type=snapshot_storage.forecaster_type,
                cutoff_date=end_date,
                data_window_start=start_date.isoformat(),
                data_window_end=end_date.isoformat(),
                symbols=main_artifacts.available_symbols,
                config=config,
                train_loss=main_artifacts.train_loss,
                val_loss=main_artifacts.val_loss,
                best_epoch=main_artifacts.best_epoch,
                stopped_epoch=main_artifacts.stopped_epoch,
                config_symbols_hash=end_snap_digest,
            )
            persist_forecaster_snapshot(
                snapshot_storage=snapshot_storage,
                cutoff_date=end_date,
                snapshot_digest=end_snap_digest,
                model=main_artifacts.model,
                feature_scaler=main_artifacts.feature_scaler,
                config=config,
                metadata=snapshot_metadata,
                train_loss=main_artifacts.train_loss,
                val_loss=main_artifacts.val_loss,
                snapshot_hf_repo=snapshot_hf_repo,
                log_prefix=log_prefix,
            )
        else:
            # Snapshots-only path: cannot recreate the end-window
            # snapshot without retraining main. Log loudly so the
            # operator can choose to delete the cached main version
            # and rerun if they want this snapshot back.
            logger.warning(
                f"{log_prefix} End-of-window snapshot for {end_date} is "
                f"missing and main is cached; skipping. Delete the cached "
                f"main version and rerun /train/lstm to recreate it."
            )

    logger.info(f"{log_prefix} Backfilling historical snapshots...")
    _backfill_lstm_snapshots(
        symbols,
        config,
        start_date,
        end_date,
        snapshot_storage,
        policy=policy,
    )


def _backfill_lstm_snapshots(
    symbols: list[str],
    config: LSTMConfig,
    start_date: date,
    end_date: date,
    snapshot_storage: SnapshotLocalStorage,
    *,
    policy: StoragePolicy | None = None,
) -> None:
    """Backfill LSTM snapshots for the RL walk-forward window.

    RL year Y needs ``snapshot-(Y-1)-12-31``; the earliest snapshot is
    ``(start_year-1)-12-31``. We extend the price window back by
    ``bootstrap_years`` so the earliest snapshot still has enough
    history to train. Prices are loaded ONCE for the extended window
    and filtered incrementally per cutoff.

    Existence checks (``check_hf``) and policy semantics: see
    :func:`_run_lstm_snapshot_phase` -- same rules apply.
    """
    if policy is None:
        policy = get_storage_policy()
    start_year = start_date.year
    end_year = end_date.year
    bootstrap_years = 4
    snapshot_hf_repo = snapshot_storage._get_hf_repo()
    check_hf = _resolve_check_hf(snapshot_storage=snapshot_storage, policy=policy)

    # RL year Y needs snapshot-(Y-1)-12-31.  Create from (start_year-1) onward.
    first_snapshot_year = start_year - 1
    snapshot_data_start = date(first_snapshot_year - bootstrap_years, 1, 1)

    snapshots_needed = []
    for year in range(first_snapshot_year, end_year):
        cutoff_date = date(year, 12, 31)
        backfill_digest = compute_model_hash(
            snapshot_storage.forecaster_type,
            snapshot_data_start,
            cutoff_date,
            symbols,
            config.to_dict(),
        )
        if not snapshot_storage.snapshot_exists_anywhere(
            cutoff_date,
            backfill_digest,
            check_hf=check_hf,
        ):
            snapshots_needed.append(cutoff_date)

    if not snapshots_needed:
        logger.info("[LSTM Backfill] All snapshots already exist, nothing to do")
        return

    logger.info(
        f"[LSTM Backfill] Need to create {len(snapshots_needed)} "
        f"snapshots: {snapshots_needed}"
    )

    # Load prices ONCE for extended window (covers bootstrap for earliest snapshot)
    logger.info(
        f"[LSTM Backfill] Loading prices from {snapshot_data_start} to {end_date}..."
    )
    t0 = time.time()
    prices_full = load_prices_yfinance(symbols, snapshot_data_start, end_date)
    t_prices = time.time() - t0
    logger.info(
        f"[LSTM Backfill] Loaded prices for {len(prices_full)} symbols in "
        f"{t_prices:.1f}s"
    )

    if len(prices_full) == 0:
        logger.warning("[LSTM Backfill] No price data loaded, cannot create snapshots")
        return

    # Train each snapshot using filtered prices
    for cutoff_date in snapshots_needed:
        logger.info(f"[LSTM Backfill] Training snapshot for cutoff {cutoff_date}")
        t0 = time.time()

        # Filter prices to cutoff (no re-download!)
        prices = _filter_prices_by_cutoff(prices_full, cutoff_date)
        if len(prices) == 0:
            logger.warning(
                f"[LSTM Backfill] No price data for cutoff {cutoff_date}, skipping"
            )
            continue

        dataset = build_dataset(prices, config)
        if len(dataset.X) == 0:
            logger.warning(
                f"[LSTM Backfill] Empty dataset for cutoff {cutoff_date}, skipping"
            )
            continue

        result = train_model_pytorch(
            dataset.X, dataset.y, dataset.feature_scaler, config
        )

        backfill_digest = compute_model_hash(
            snapshot_storage.forecaster_type,
            snapshot_data_start,
            cutoff_date,
            symbols,
            config.to_dict(),
        )

        metadata = create_snapshot_metadata(
            forecaster_type=snapshot_storage.forecaster_type,
            cutoff_date=cutoff_date,
            data_window_start=snapshot_data_start.isoformat(),
            data_window_end=cutoff_date.isoformat(),
            symbols=list(prices.keys()),
            config=config,
            train_loss=result.train_loss,
            val_loss=result.val_loss,
            best_epoch=result.best_epoch,
            stopped_epoch=result.stopped_epoch,
            config_symbols_hash=backfill_digest,
        )

        persist_forecaster_snapshot(
            snapshot_storage=snapshot_storage,
            cutoff_date=cutoff_date,
            snapshot_digest=backfill_digest,
            model=result.model,
            feature_scaler=result.feature_scaler,
            config=config,
            metadata=metadata,
            train_loss=result.train_loss,
            val_loss=result.val_loss,
            snapshot_hf_repo=snapshot_hf_repo,
            log_prefix="[LSTM Backfill]",
        )
        logger.info(
            f"[LSTM Backfill] Persist finished for {cutoff_date} in "
            f"{time.time() - t0:.1f}s"
        )

        # Memory cleanup after each snapshot to prevent accumulation
        del dataset, result, prices, metadata
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# PatchTST snapshot phase
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PatchTSTMainTrainingArtifacts:
    """In-memory PatchTST main-training outputs piggybacked into the
    end-of-window snapshot. Mirror of :class:`_LSTMMainTrainingArtifacts`."""

    model: Any
    feature_scaler: Any
    train_loss: float
    val_loss: float
    best_epoch: int
    stopped_epoch: int
    available_symbols: list[str]


def _run_patchtst_snapshot_phase(
    *,
    train_window: tuple[date, date],
    symbols: list[str],
    config: PatchTSTConfig,
    snapshot_storage: SnapshotLocalStorage,
    main_artifacts: _PatchTSTMainTrainingArtifacts | None,
    policy: StoragePolicy | None = None,
    log_prefix: str = "[PatchTST]",
) -> None:
    """PatchTST snapshot phase. Mirror of :func:`_run_lstm_snapshot_phase`
    -- same warn-and-skip rule for the missing end-window snapshot in
    the snapshots-only path, same policy semantics for existence
    checks, same ``StoragePolicyError`` propagation contract."""
    if policy is None:
        policy = get_storage_policy()
    start_date, end_date = train_window
    snapshot_forecaster_type = snapshot_storage.forecaster_type
    snapshot_hf_repo = snapshot_storage._get_hf_repo()
    check_hf = _resolve_check_hf(snapshot_storage=snapshot_storage, policy=policy)

    end_snap_digest = compute_model_hash(
        snapshot_forecaster_type,
        start_date,
        end_date,
        symbols,
        config.to_dict(),
    )

    end_window_present = snapshot_storage.snapshot_exists_anywhere(
        end_date,
        end_snap_digest,
        check_hf=check_hf,
    )

    if not end_window_present:
        if main_artifacts is not None:
            snapshot_metadata = create_snapshot_metadata(
                forecaster_type=snapshot_forecaster_type,
                cutoff_date=end_date,
                data_window_start=start_date.isoformat(),
                data_window_end=end_date.isoformat(),
                symbols=main_artifacts.available_symbols,
                config=config,
                train_loss=main_artifacts.train_loss,
                val_loss=main_artifacts.val_loss,
                best_epoch=main_artifacts.best_epoch,
                stopped_epoch=main_artifacts.stopped_epoch,
                config_symbols_hash=end_snap_digest,
            )
            persist_forecaster_snapshot(
                snapshot_storage=snapshot_storage,
                cutoff_date=end_date,
                snapshot_digest=end_snap_digest,
                model=main_artifacts.model,
                feature_scaler=main_artifacts.feature_scaler,
                config=config,
                metadata=snapshot_metadata,
                train_loss=main_artifacts.train_loss,
                val_loss=main_artifacts.val_loss,
                snapshot_hf_repo=snapshot_hf_repo,
                log_prefix=log_prefix,
            )
        else:
            logger.warning(
                f"{log_prefix} End-of-window snapshot for {end_date} is "
                f"missing and main is cached; skipping. Delete the cached "
                f"main version and rerun /train/patchtst (or "
                f"/train/patchtst/india) to recreate it."
            )

    logger.info(f"{log_prefix} Backfilling historical snapshots...")
    _backfill_patchtst_snapshots(
        symbols,
        config,
        start_date,
        end_date,
        snapshot_storage,
        log_prefix=log_prefix,
        policy=policy,
    )


def _backfill_patchtst_snapshots(
    symbols: list[str],
    config: PatchTSTConfig,
    start_date: date,
    end_date: date,
    snapshot_storage: SnapshotLocalStorage,
    log_prefix: str = "[PatchTST Backfill]",
    *,
    policy: StoragePolicy | None = None,
) -> None:
    """Backfill PatchTST snapshots for the RL walk-forward window.

    Mirror of :func:`_backfill_lstm_snapshots` -- same window math,
    same digest formula, same ``policy`` semantics. Adds OHLCV-specific
    ``align_multivariate_data`` + ``patchtst_build_dataset`` plumbing
    per cutoff.
    """
    if policy is None:
        policy = get_storage_policy()
    backfill_prefix = (
        f"{log_prefix} Backfill" if "Backfill" not in log_prefix else log_prefix
    )
    start_year = start_date.year
    end_year = end_date.year
    bootstrap_years = 4
    snapshot_hf_repo = snapshot_storage._get_hf_repo()
    check_hf = _resolve_check_hf(snapshot_storage=snapshot_storage, policy=policy)

    first_snapshot_year = start_year - 1
    snapshot_data_start = date(first_snapshot_year - bootstrap_years, 1, 1)

    snapshots_needed = []
    for year in range(first_snapshot_year, end_year):
        cutoff_date = date(year, 12, 31)
        backfill_digest = compute_model_hash(
            snapshot_storage.forecaster_type,
            snapshot_data_start,
            cutoff_date,
            symbols,
            config.to_dict(),
        )
        if not snapshot_storage.snapshot_exists_anywhere(
            cutoff_date,
            backfill_digest,
            check_hf=check_hf,
        ):
            snapshots_needed.append(cutoff_date)

    if not snapshots_needed:
        logger.info(f"[{backfill_prefix}] All snapshots already exist, nothing to do")
        return

    logger.info(
        f"[{backfill_prefix}] Need to create {len(snapshots_needed)} "
        f"snapshots: {snapshots_needed}"
    )

    logger.info(
        f"[{backfill_prefix}] Loading prices from {snapshot_data_start} "
        f"to {end_date}..."
    )
    t0 = time.time()
    prices_full = patchtst_load_prices(symbols, snapshot_data_start, end_date)
    t_prices = time.time() - t0
    logger.info(
        f"[{backfill_prefix}] Loaded prices for {len(prices_full)} symbols "
        f"in {t_prices:.1f}s"
    )

    if len(prices_full) == 0:
        logger.warning(
            f"[{backfill_prefix}] No price data loaded, cannot create snapshots"
        )
        return

    snapshot_forecaster_type = snapshot_storage.forecaster_type

    for cutoff_date in snapshots_needed:
        logger.info(f"[{backfill_prefix}] Training snapshot for cutoff {cutoff_date}")
        t0 = time.time()

        prices = _filter_prices_by_cutoff(prices_full, cutoff_date)
        if len(prices) == 0:
            logger.warning(
                f"[{backfill_prefix}] No price data for cutoff {cutoff_date}, skipping"
            )
            continue

        aligned_features = align_multivariate_data(prices, config)

        if len(aligned_features) == 0:
            logger.warning(
                f"[{backfill_prefix}] No aligned features for cutoff {cutoff_date}, skipping"
            )
            continue

        dataset = patchtst_build_dataset(aligned_features, prices, config)
        if len(dataset.X) == 0:
            logger.warning(
                f"[{backfill_prefix}] Empty dataset for cutoff {cutoff_date}, skipping"
            )
            continue

        result = patchtst_train_model(
            dataset.X,
            dataset.y,
            dataset.feature_scaler,
            config,
            anchor_dates=dataset.anchor_dates,
            sample_symbols=dataset.symbols,
        )

        backfill_digest = compute_model_hash(
            snapshot_forecaster_type,
            snapshot_data_start,
            cutoff_date,
            symbols,
            config.to_dict(),
        )

        metadata = create_snapshot_metadata(
            forecaster_type=snapshot_forecaster_type,
            cutoff_date=cutoff_date,
            data_window_start=snapshot_data_start.isoformat(),
            data_window_end=cutoff_date.isoformat(),
            symbols=list(prices.keys()),
            config=config,
            train_loss=result.train_loss,
            val_loss=result.val_loss,
            best_epoch=result.best_epoch,
            stopped_epoch=result.stopped_epoch,
            config_symbols_hash=backfill_digest,
        )

        persist_forecaster_snapshot(
            snapshot_storage=snapshot_storage,
            cutoff_date=cutoff_date,
            snapshot_digest=backfill_digest,
            model=result.model,
            feature_scaler=result.feature_scaler,
            config=config,
            metadata=metadata,
            train_loss=result.train_loss,
            val_loss=result.val_loss,
            snapshot_hf_repo=snapshot_hf_repo,
            log_prefix=backfill_prefix,
        )
        logger.info(
            f"[{backfill_prefix}] Persist finished for {cutoff_date} in "
            f"{time.time() - t0:.1f}s"
        )

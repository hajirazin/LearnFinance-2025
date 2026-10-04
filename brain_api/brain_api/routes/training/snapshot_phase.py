"""Forecaster snapshot phase shared by LSTM + PatchTST training routes.

Houses the snapshot-phase helpers that the main-training background
runners and the snapshots-only background runners both depend on:

* Per-family ``_run_*_snapshot_phase`` backfills December 31 snapshots
  only. An expected digest already on disk or Hugging Face is copied
  to the other side and is not trained.
* Per-family ``_backfill_*_snapshots`` trains the cutoffs whose parity
  is ``train`` (or whose download was refused). ``policy`` is accepted
  and ignored: annual parity does not read ``STORAGE_BACKEND``.

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
from datetime import date

import torch

from brain_api.core.forecaster_snapshot_identity import (
    annual_snapshot_cutoffs,
    annual_snapshot_parity,
)
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
from brain_api.core.version import compute_snapshot_identity_hash
from brain_api.routes.training.snapshot_persist import persist_forecaster_snapshot
from brain_api.routes.training.snapshot_phase_filters import (
    _filter_prices_by_cutoff as _filter_prices_by_cutoff,
)
from brain_api.routes.training.snapshot_phase_filters import (
    _filter_signals_by_cutoff as _filter_signals_by_cutoff,
)
from brain_api.storage.forecaster_snapshots import (
    SnapshotLocalStorage,
    create_snapshot_metadata,
)
from brain_api.storage.policy import StoragePolicy

logger = logging.getLogger(__name__)


def _cutoffs_requiring_training(
    snapshot_storage: SnapshotLocalStorage,
    config_dict: dict,
    start_date: date,
    end_date: date,
) -> list[date]:
    """Copy one-sided annual digests and return cutoffs that still need training.

    Upload failures propagate. A download that returns ``False`` (unhealthy
    branch or failed install) is trained. ``STORAGE_BACKEND`` is not read.
    """
    uploads: list[tuple[date, str]] = []
    downloads: list[tuple[date, str]] = []
    trains: list[date] = []
    for cutoff_date in annual_snapshot_cutoffs(start_date, end_date):
        digest = compute_snapshot_identity_hash(
            snapshot_storage.forecaster_type,
            cutoff_date,
            config_dict,
        )
        action = annual_snapshot_parity(snapshot_storage, cutoff_date, digest)
        if action == "upload":
            uploads.append((cutoff_date, digest))
        elif action == "download":
            downloads.append((cutoff_date, digest))
        elif action == "train":
            trains.append(cutoff_date)

    for cutoff_date, digest in uploads:
        uploaded = snapshot_storage.upload_snapshot_to_hf(cutoff_date, digest)
        if not uploaded:
            raise RuntimeError(
                f"Failed to upload {snapshot_storage.forecaster_type} "
                f"snapshot {cutoff_date} ({digest})"
            )
    for cutoff_date, digest in downloads:
        if not snapshot_storage.download_snapshot_from_hf(cutoff_date, digest):
            trains.append(cutoff_date)
    return trains


# ---------------------------------------------------------------------------
# LSTM snapshot phase
# ---------------------------------------------------------------------------


def _run_lstm_snapshot_phase(
    *,
    train_window: tuple[date, date],
    symbols: list[str],
    config: LSTMConfig,
    snapshot_storage: SnapshotLocalStorage,
    policy: StoragePolicy | None = None,
    log_prefix: str = "[LSTM]",
) -> None:
    """Backfill December 31 LSTM snapshots for the training window.

    ``policy`` is ignored. A digest on either side is copied; neither
    side is trained.
    """
    del policy
    start_date, end_date = train_window
    logger.info(f"{log_prefix} Backfilling annual snapshots...")
    _backfill_lstm_snapshots(
        symbols,
        config,
        start_date,
        end_date,
        snapshot_storage,
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
    """Backfill LSTM December 31 snapshots for the RL walk-forward window.

    RL year Y needs ``snapshot-(Y-1)-12-31``; the earliest snapshot is
    ``(start_year-1)-12-31``. A cutoff is included only when that
    December 31 is on or before ``end_date``. We extend the price window
    back by ``bootstrap_years`` so the earliest snapshot still has enough
    history to train. Prices are loaded ONCE for the extended window
    and filtered incrementally per cutoff.

    ``policy`` is ignored. One-sided digests are copied; only cutoffs
    with neither side (or a refused download) are trained.
    """
    del policy
    bootstrap_years = 4
    snapshot_hf_repo = snapshot_storage._get_hf_repo()
    cutoffs = annual_snapshot_cutoffs(start_date, end_date)
    first_snapshot_year = cutoffs[0].year if cutoffs else start_date.year - 1
    snapshot_data_start = date(first_snapshot_year - bootstrap_years, 1, 1)

    snapshots_needed = _cutoffs_requiring_training(
        snapshot_storage,
        config.to_dict(),
        start_date,
        end_date,
    )

    if not snapshots_needed:
        logger.info("[LSTM Backfill] All annual snapshots are ready, nothing to train")
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

        backfill_digest = compute_snapshot_identity_hash(
            snapshot_storage.forecaster_type,
            cutoff_date,
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


def _run_patchtst_snapshot_phase(
    *,
    train_window: tuple[date, date],
    symbols: list[str],
    config: PatchTSTConfig,
    snapshot_storage: SnapshotLocalStorage,
    policy: StoragePolicy | None = None,
    log_prefix: str = "[PatchTST]",
) -> None:
    """Backfill December 31 PatchTST snapshots. Mirror of the LSTM phase."""
    del policy
    start_date, end_date = train_window
    logger.info(f"{log_prefix} Backfilling annual snapshots...")
    _backfill_patchtst_snapshots(
        symbols,
        config,
        start_date,
        end_date,
        snapshot_storage,
        log_prefix=log_prefix,
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
    """Backfill PatchTST December 31 snapshots for the RL walk-forward window.

    Mirror of :func:`_backfill_lstm_snapshots` -- same cutoff math,
    same digest formula, same parity copy-or-train rule. Adds
    OHLCV-specific ``align_multivariate_data`` + ``patchtst_build_dataset``
    plumbing per cutoff. ``policy`` is ignored.
    """
    del policy
    backfill_prefix = (
        f"{log_prefix} Backfill" if "Backfill" not in log_prefix else log_prefix
    )
    bootstrap_years = 4
    snapshot_hf_repo = snapshot_storage._get_hf_repo()
    cutoffs = annual_snapshot_cutoffs(start_date, end_date)
    first_snapshot_year = cutoffs[0].year if cutoffs else start_date.year - 1
    snapshot_data_start = date(first_snapshot_year - bootstrap_years, 1, 1)

    snapshots_needed = _cutoffs_requiring_training(
        snapshot_storage,
        config.to_dict(),
        start_date,
        end_date,
    )

    if not snapshots_needed:
        logger.info(
            f"[{backfill_prefix}] All annual snapshots are ready, nothing to train"
        )
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

        backfill_digest = compute_snapshot_identity_hash(
            snapshot_forecaster_type,
            cutoff_date,
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

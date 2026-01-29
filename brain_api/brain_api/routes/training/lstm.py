"""LSTM training endpoint."""

import gc
import logging
import time
from datetime import date

import pandas as pd
import torch
from fastapi import APIRouter, Depends, Query

from brain_api.core.config import (
    get_hf_lstm_model_repo,
    get_storage_backend,
    resolve_training_window,
)
from brain_api.core.lstm import (
    LSTMConfig,
    build_dataset,
    compute_version,
    evaluate_for_promotion,
    load_prices_yfinance,
    train_model_pytorch,
)
from brain_api.storage.forecaster_snapshots import (
    SnapshotLocalStorage,
    create_snapshot_metadata,
)
from brain_api.storage.local import LocalModelStorage, create_metadata

from .dependencies import (
    DatasetBuilder,
    PriceLoader,
    Trainer,
    get_config,
    get_dataset_builder,
    get_lstm_training_symbols,
    get_price_loader,
    get_storage,
    get_trainer,
)
from .helpers import get_prior_version_info
from .models import LSTMTrainResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/lstm", response_model=LSTMTrainResponse)
def train_lstm(
    skip_snapshot: bool = Query(
        False,
        description="Skip saving snapshot (by default saves snapshot for current + all historical years)",
    ),
    storage: LocalModelStorage = Depends(get_storage),
    symbols: list[str] = Depends(get_lstm_training_symbols),
    config: LSTMConfig = Depends(get_config),
    price_loader: PriceLoader = Depends(get_price_loader),
    dataset_builder: DatasetBuilder = Depends(get_dataset_builder),
    trainer: Trainer = Depends(get_trainer),
) -> LSTMTrainResponse:
    """Train the shared LSTM model for weekly return prediction.

    The model predicts weekly returns (Mon open → Fri close) to align with
    the RL agent's weekly decision horizon. This naturally handles holidays
    as a "week" is simply the first-to-last trading day of each ISO week.

    Uses API config for data window (default: last 15 years).
    Fetches price data from yfinance for the configured training universe.
    Universe is controlled by LSTM_TRAIN_UNIVERSE env var (default: "halal").
    Writes versioned artifacts and promotes if evaluation passes.

    By default, also saves snapshots for all historical years (for walk-forward
    forecast generation in RL training). Use skip_snapshot=true to disable.

    Args:
        skip_snapshot: If True, skips saving snapshots. By default (False),
                      saves snapshot for current training window + all historical years.

    Returns:
        Training result including version, metrics, and promotion status.
    """
    # Resolve window from API config
    start_date, end_date = resolve_training_window()
    logger.info(f"[LSTM] Starting training for {len(symbols)} symbols")
    logger.info(f"[LSTM] Data window: {start_date} to {end_date}")
    logger.info(f"[LSTM] Symbols: {symbols}")

    # Compute deterministic version
    version = compute_version(start_date, end_date, symbols, config)
    logger.info(f"[LSTM] Computed version: {version}")

    # Check if this version already exists (idempotent)
    if storage.version_exists(version):
        logger.info(
            f"[LSTM] Version {version} already exists (idempotent), returning cached result"
        )
        # Return existing metadata
        existing_metadata = storage.read_metadata(version)
        if existing_metadata:
            return LSTMTrainResponse(
                version=version,
                data_window_start=existing_metadata["data_window"]["start"],
                data_window_end=existing_metadata["data_window"]["end"],
                metrics=existing_metadata["metrics"],
                promoted=existing_metadata["promoted"],
                prior_version=existing_metadata.get("prior_version"),
            )

    # Load price data
    logger.info(f"[LSTM] Loading price data for {len(symbols)} symbols...")
    t0 = time.time()
    prices = price_loader(symbols, start_date, end_date)
    t_prices = time.time() - t0
    logger.info(
        f"[LSTM] Loaded prices for {len(prices)}/{len(symbols)} symbols in {t_prices:.1f}s"
    )

    if len(prices) == 0:
        logger.error("[LSTM] No price data loaded - cannot train model")
        raise ValueError("No price data available for training")

    # Build dataset (returns DatasetResult with X, y (weekly returns), feature_scaler)
    logger.info("[LSTM] Building dataset...")
    t0 = time.time()
    dataset = dataset_builder(prices, config)
    t_dataset = time.time() - t0
    logger.info(f"[LSTM] Dataset built in {t_dataset:.1f}s: {len(dataset.X)} samples")

    if len(dataset.X) == 0:
        logger.error("[LSTM] Dataset is empty - cannot train model")
        raise ValueError("No training samples could be built from price data")

    # Train model
    logger.info("[LSTM] Starting model training...")
    t0 = time.time()
    result = trainer(
        dataset.X,
        dataset.y,
        dataset.feature_scaler,
        config,
    )
    t_train = time.time() - t0
    logger.info(f"[LSTM] Training complete in {t_train:.1f}s")
    logger.info(
        f"[LSTM] Metrics: train_loss={result.train_loss:.6f}, val_loss={result.val_loss:.6f}, baseline={result.baseline_loss:.6f}"
    )

    # Get prior version info for promotion decision (checks local, then HF if needed)
    from brain_api.storage.huggingface import HuggingFaceModelStorage

    hf_model_repo = get_hf_lstm_model_repo()
    prior_info = get_prior_version_info(
        local_storage=storage,
        hf_storage_class=HuggingFaceModelStorage,
        hf_model_repo=hf_model_repo,
    )
    prior_version = prior_info.version
    prior_val_loss = prior_info.val_loss

    if prior_version:
        logger.info(
            f"[LSTM] Prior version: {prior_version}, val_loss: {prior_val_loss}"
        )
    else:
        logger.info("[LSTM] No prior version exists (first model)")

    # Decide on promotion
    promoted = evaluate_for_promotion(
        val_loss=result.val_loss,
        prior_val_loss=prior_val_loss,
    )
    logger.info(
        f"[LSTM] Promotion decision: {'PROMOTED' if promoted else 'NOT promoted'}"
    )

    # Create metadata
    metadata = create_metadata(
        version=version,
        data_window_start=start_date.isoformat(),
        data_window_end=end_date.isoformat(),
        symbols=symbols,
        config=config,
        train_loss=result.train_loss,
        val_loss=result.val_loss,
        baseline_loss=result.baseline_loss,
        promoted=promoted,
        prior_version=prior_version,
    )

    # Write artifacts locally
    logger.info(f"[LSTM] Writing artifacts for version {version}...")
    storage.write_artifacts(
        version=version,
        model=result.model,
        feature_scaler=result.feature_scaler,
        config=config,
        metadata=metadata,
    )
    logger.info("[LSTM] Artifacts written successfully")

    # Promote if passed evaluation, or if this is the first model (so inference has something)
    if promoted or prior_version is None:
        storage.promote_version(version)
        logger.info(f"[LSTM] Version {version} promoted to current")

    # Optionally push to HuggingFace Hub
    hf_repo = None
    hf_url = None
    storage_backend = get_storage_backend()

    if storage_backend == "hf" and hf_model_repo:
        try:
            from brain_api.storage.huggingface import HuggingFaceModelStorage

            hf_storage = HuggingFaceModelStorage(repo_id=hf_model_repo)

            # Check if HF main branch has a version (might be empty even if local has one)
            hf_has_main = hf_storage.get_current_version() is not None

            # Promote to main if: passed promotion check OR HF main is empty (first upload)
            should_make_current = promoted or not hf_has_main
            logger.info(
                f"[LSTM] HF upload: promoted={promoted}, hf_has_main={hf_has_main}, "
                f"make_current={should_make_current}"
            )

            hf_info = hf_storage.upload_model(
                version=version,
                model=result.model,
                feature_scaler=result.feature_scaler,
                config=config,
                metadata=metadata,
                make_current=should_make_current,
            )
            hf_repo = hf_info.repo_id
            hf_url = f"https://huggingface.co/{hf_info.repo_id}/tree/{version}"
            logger.info(f"[LSTM] Model uploaded to HuggingFace: {hf_url}")
        except Exception as e:
            logger.error(f"Failed to upload model to HuggingFace: {e}")
            # Don't fail the training request if HF upload fails

    # Save snapshots (unless skip_snapshot=True)
    if not skip_snapshot:
        snapshot_storage = SnapshotLocalStorage("lstm")
        check_hf = storage_backend == "hf"

        # Save snapshot for current training window
        if not snapshot_storage.snapshot_exists_anywhere(end_date, check_hf=check_hf):
            snapshot_metadata = create_snapshot_metadata(
                forecaster_type="lstm",
                cutoff_date=end_date,
                data_window_start=start_date.isoformat(),
                data_window_end=end_date.isoformat(),
                symbols=list(prices.keys()),
                config=config,
                train_loss=result.train_loss,
                val_loss=result.val_loss,
            )
            snapshot_storage.write_snapshot(
                cutoff_date=end_date,
                model=result.model,
                feature_scaler=result.feature_scaler,
                config=config,
                metadata=snapshot_metadata,
            )
            logger.info(f"[LSTM] Saved snapshot for cutoff {end_date}")

            # Upload to HuggingFace if in HF mode
            if storage_backend == "hf":
                try:
                    snapshot_storage.upload_snapshot_to_hf(end_date)
                    logger.info(f"[LSTM] Uploaded snapshot {end_date} to HuggingFace")
                except Exception as e:
                    logger.error(f"[LSTM] Failed to upload snapshot to HF: {e}")

        # Also backfill all historical snapshots
        logger.info("[LSTM] Backfilling historical snapshots...")
        _backfill_lstm_snapshots(
            symbols, config, start_date, end_date, snapshot_storage, storage_backend
        )

    return LSTMTrainResponse(
        version=version,
        data_window_start=start_date.isoformat(),
        data_window_end=end_date.isoformat(),
        metrics={
            "train_loss": result.train_loss,
            "val_loss": result.val_loss,
            "baseline_loss": result.baseline_loss,
        },
        promoted=promoted,
        prior_version=prior_version,
        hf_repo=hf_repo,
        hf_url=hf_url,
    )


def _filter_prices_by_cutoff(
    prices: dict[str, pd.DataFrame],
    cutoff_date: date,
) -> dict[str, pd.DataFrame]:
    """Filter price DataFrames to include only data up to cutoff_date.

    Args:
        prices: Dict mapping symbol -> DataFrame with DatetimeIndex
        cutoff_date: Include data up to and including this date

    Returns:
        Filtered dict with same structure, excluding symbols with no data after filtering
    """
    cutoff_ts = pd.Timestamp(cutoff_date)
    return {
        symbol: df[df.index <= cutoff_ts].copy()
        for symbol, df in prices.items()
        if len(df[df.index <= cutoff_ts]) > 0
    }


def _backfill_lstm_snapshots(
    symbols: list[str],
    config: LSTMConfig,
    start_date: date,
    end_date: date,
    snapshot_storage: SnapshotLocalStorage,
    storage_backend: str = "local",
) -> None:
    """Backfill LSTM snapshots for all historical years.

    For each year from (start_year + 4) to end_year-1, trains a snapshot
    on data up to Dec 31 of that year. Uses 4-year bootstrap period.

    Optimization: Loads prices ONCE for the full window and filters incrementally
    by year instead of re-downloading for each snapshot.

    Args:
        symbols: List of stock symbols
        config: LSTM configuration
        start_date: Training data start date
        end_date: Training data end date
        snapshot_storage: Storage instance
        storage_backend: "local" or "hf" - if "hf", uploads to HuggingFace
    """
    start_year = start_date.year
    end_year = end_date.year
    bootstrap_years = 4
    check_hf = storage_backend == "hf"

    # Check if any snapshots need to be created (check HF too if in HF mode)
    snapshots_needed = []
    for year in range(start_year + bootstrap_years, end_year):
        cutoff_date = date(year, 12, 31)
        if not snapshot_storage.snapshot_exists_anywhere(
            cutoff_date, check_hf=check_hf
        ):
            snapshots_needed.append(cutoff_date)

    if not snapshots_needed:
        logger.info("[LSTM Backfill] All snapshots already exist, nothing to do")
        return

    logger.info(
        f"[LSTM Backfill] Need to create {len(snapshots_needed)} snapshots: {snapshots_needed}"
    )

    # Load prices ONCE for full window
    logger.info("[LSTM Backfill] Loading prices for full window (single download)...")
    t0 = time.time()
    prices_full = load_prices_yfinance(symbols, start_date, end_date)
    t_prices = time.time() - t0
    logger.info(
        f"[LSTM Backfill] Loaded prices for {len(prices_full)} symbols in {t_prices:.1f}s"
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

        metadata = create_snapshot_metadata(
            forecaster_type="lstm",
            cutoff_date=cutoff_date,
            data_window_start=start_date.isoformat(),
            data_window_end=cutoff_date.isoformat(),
            symbols=list(prices.keys()),
            config=config,
            train_loss=result.train_loss,
            val_loss=result.val_loss,
        )

        snapshot_storage.write_snapshot(
            cutoff_date=cutoff_date,
            model=result.model,
            feature_scaler=result.feature_scaler,
            config=config,
            metadata=metadata,
        )
        logger.info(
            f"[LSTM Backfill] Saved snapshot for {cutoff_date} in {time.time() - t0:.1f}s"
        )

        # Upload to HuggingFace if in HF mode
        if storage_backend == "hf":
            try:
                snapshot_storage.upload_snapshot_to_hf(cutoff_date)
                logger.info(
                    f"[LSTM Backfill] Uploaded snapshot {cutoff_date} to HuggingFace"
                )
            except Exception as e:
                logger.error(f"[LSTM Backfill] Failed to upload snapshot to HF: {e}")

        # Memory cleanup after each snapshot to prevent accumulation
        del dataset, result, prices, metadata
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

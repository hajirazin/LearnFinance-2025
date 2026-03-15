"""PatchTST training endpoint."""

import gc
import logging
import threading
import time
from collections.abc import Callable
from datetime import date

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, Query
from fastapi.responses import JSONResponse

from brain_api.core.config import (
    get_hf_patchtst_model_repo,
    get_storage_backend,
    resolve_training_window,
)
from brain_api.core.patchtst import PatchTSTConfig, align_multivariate_data
from brain_api.core.patchtst import (
    build_dataset as patchtst_build_dataset,
)
from brain_api.core.patchtst import (
    compute_version as patchtst_compute_version,
)
from brain_api.core.patchtst import (
    evaluate_for_promotion as patchtst_evaluate_for_promotion,
)
from brain_api.core.patchtst import (
    load_prices_yfinance as patchtst_load_prices,
)
from brain_api.core.patchtst import (
    train_model_pytorch as patchtst_train_model,
)
from brain_api.core.training_utils import TrainingCancelledError
from brain_api.storage.forecaster_snapshots import (
    SnapshotLocalStorage,
    create_snapshot_metadata,
)
from brain_api.storage.local import PatchTSTModelStorage, create_patchtst_metadata

from .dependencies import (
    PatchTSTDatasetBuilder,
    PatchTSTPriceLoader,
    PatchTSTTrainer,
    get_forecaster_training_symbols,
    get_patchtst_config,
    get_patchtst_dataset_builder,
    get_patchtst_price_loader,
    get_patchtst_storage,
    get_patchtst_trainer,
)
from .helpers import get_prior_version_info
from .job_registry import (
    cancel_job,
    complete_job,
    fail_job,
    get_or_create_job,
    update_progress,
)
from .models import PatchTSTTrainResponse, TrainingJobResponse

router = APIRouter()
logger = logging.getLogger(__name__)


def _train_patchtst_core(
    symbols: list[str],
    storage: PatchTSTModelStorage,
    hf_storage_class: type,
    hf_model_repo_getter: Callable[[], str | None],
    snapshot_forecaster_type: str,
    skip_snapshot: bool,
    config: PatchTSTConfig,
    price_loader: PatchTSTPriceLoader,
    dataset_builder: PatchTSTDatasetBuilder,
    trainer: PatchTSTTrainer,
    log_prefix: str = "[PatchTST]",
    shutdown_event: threading.Event | None = None,
    job_id: str | None = None,
) -> PatchTSTTrainResponse:
    """Core PatchTST training logic shared by US and India endpoints.

    Handles: version check -> load prices -> align -> build dataset -> train ->
    evaluate promotion -> write artifacts -> HF upload -> snapshot backfill.

    Args:
        symbols: Stock symbols to train on.
        storage: Local model storage instance.
        hf_storage_class: HuggingFace storage class for this market.
        hf_model_repo_getter: Callable returning HF repo ID.
        snapshot_forecaster_type: "patchtst" or "patchtst_india".
        skip_snapshot: Skip saving snapshots.
        config: PatchTST training configuration.
        price_loader: Function to load price data.
        dataset_builder: Function to build datasets.
        trainer: Function to train the model.
        log_prefix: Logging prefix string.

    Returns:
        PatchTSTTrainResponse with training results.
    """
    start_date, end_date = resolve_training_window()
    logger.info(f"{log_prefix} Starting training for {len(symbols)} symbols")
    logger.info(f"{log_prefix} Data window: {start_date} to {end_date}")
    logger.info(f"{log_prefix} Symbols: {symbols}")
    logger.info(
        f"{log_prefix} Config: {config.num_input_channels} channels, {config.epochs} epochs"
    )

    version = patchtst_compute_version(start_date, end_date, symbols, config)
    logger.info(f"{log_prefix} Computed version: {version}")

    if storage.version_exists(version):
        logger.info(
            f"{log_prefix} Version {version} already exists (idempotent), returning cached result"
        )
        existing_metadata = storage.read_metadata(version)
        if existing_metadata:
            return PatchTSTTrainResponse(
                version=version,
                data_window_start=existing_metadata["data_window"]["start"],
                data_window_end=existing_metadata["data_window"]["end"],
                metrics=existing_metadata["metrics"],
                promoted=existing_metadata["promoted"],
                prior_version=existing_metadata.get("prior_version"),
                num_input_channels=config.num_input_channels,
                signals_used=["ohlcv"],
            )

    if job_id:
        update_progress(job_id, {"phase": "loading_prices"})
    logger.info(f"{log_prefix} Loading price data for {len(symbols)} symbols...")
    t0 = time.time()
    prices = price_loader(symbols, start_date, end_date)
    t_prices = time.time() - t0
    logger.info(
        f"{log_prefix} Loaded prices for {len(prices)}/{len(symbols)} symbols in {t_prices:.1f}s"
    )

    if len(prices) == 0:
        logger.error(f"{log_prefix} No price data loaded - cannot train model")
        raise ValueError("No price data available for training")

    logger.info(f"{log_prefix} Aligning multivariate data (OHLCV only)...")
    t0 = time.time()
    aligned_features = align_multivariate_data(prices, config)
    t_align = time.time() - t0
    logger.info(
        f"{log_prefix} Aligned data for {len(aligned_features)}/{len(prices)} symbols in {t_align:.1f}s"
    )

    if len(aligned_features) == 0:
        logger.error(f"{log_prefix} No aligned features - cannot train model")
        raise ValueError("No aligned features could be built from available data")

    if job_id:
        update_progress(job_id, {"phase": "building_dataset"})
    logger.info(f"{log_prefix} Building dataset...")
    t0 = time.time()
    dataset = dataset_builder(aligned_features, prices, config)
    t_dataset = time.time() - t0
    logger.info(
        f"{log_prefix} Dataset built in {t_dataset:.1f}s: {len(dataset.X)} samples"
    )

    available_symbols = list(prices.keys())

    del aligned_features, prices

    if len(dataset.X) == 0:
        logger.error(f"{log_prefix} Dataset is empty - cannot train model")
        raise ValueError("No training samples could be built from aligned features")

    X, y, feature_scaler = dataset.X, dataset.y, dataset.feature_scaler
    del dataset
    gc.collect()

    if job_id:
        update_progress(job_id, {"phase": "training"})
    logger.info(f"{log_prefix} Starting model training...")
    t0 = time.time()
    result = trainer(X, y, feature_scaler, config, shutdown_event=shutdown_event)
    t_train = time.time() - t0
    logger.info(f"{log_prefix} Training complete in {t_train:.1f}s")
    logger.info(
        f"{log_prefix} Metrics: train_loss={result.train_loss:.6f}, val_loss={result.val_loss:.6f}, baseline={result.baseline_loss:.6f}"
    )

    hf_model_repo = hf_model_repo_getter()
    prior_info = get_prior_version_info(
        local_storage=storage,
        hf_storage_class=hf_storage_class,
        hf_model_repo=hf_model_repo,
    )
    prior_version = prior_info.version
    prior_val_loss = prior_info.val_loss

    if prior_version:
        logger.info(
            f"{log_prefix} Prior version: {prior_version}, val_loss: {prior_val_loss}"
        )
    else:
        logger.info(f"{log_prefix} No prior version exists (first model)")

    promoted = patchtst_evaluate_for_promotion(
        val_loss=result.val_loss,
        prior_val_loss=prior_val_loss,
    )
    logger.info(
        f"{log_prefix} Promotion decision: {'PROMOTED' if promoted else 'NOT promoted'}"
    )

    metadata = create_patchtst_metadata(
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

    logger.info(f"{log_prefix} Writing artifacts for version {version}...")
    storage.write_artifacts(
        version=version,
        model=result.model,
        feature_scaler=result.feature_scaler,
        config=config,
        metadata=metadata,
    )
    logger.info(f"{log_prefix} Artifacts written successfully")

    if promoted or prior_version is None:
        storage.promote_version(version)
        logger.info(f"{log_prefix} Version {version} promoted to current")

    hf_repo = None
    hf_url = None
    storage_backend = get_storage_backend()

    if storage_backend == "hf" and hf_model_repo:
        try:
            hf_storage = hf_storage_class(repo_id=hf_model_repo)
            hf_has_main = hf_storage.get_current_version() is not None
            should_make_current = promoted or not hf_has_main
            logger.info(
                f"{log_prefix} HF upload: promoted={promoted}, hf_has_main={hf_has_main}, "
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
            logger.info(f"{log_prefix} Model uploaded to HuggingFace: {hf_url}")
        except Exception as e:
            logger.error(f"{log_prefix} Failed to upload model to HuggingFace: {e}")

    if not skip_snapshot:
        snapshot_storage = SnapshotLocalStorage(snapshot_forecaster_type)
        check_hf = storage_backend == "hf"

        if not snapshot_storage.snapshot_exists_anywhere(end_date, check_hf=check_hf):
            snapshot_metadata = create_snapshot_metadata(
                forecaster_type=snapshot_forecaster_type,
                cutoff_date=end_date,
                data_window_start=start_date.isoformat(),
                data_window_end=end_date.isoformat(),
                symbols=available_symbols,
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
            logger.info(f"{log_prefix} Saved snapshot for cutoff {end_date}")

            if storage_backend == "hf":
                try:
                    snapshot_storage.upload_snapshot_to_hf(end_date)
                    logger.info(
                        f"{log_prefix} Uploaded snapshot {end_date} to HuggingFace"
                    )
                except Exception as e:
                    logger.error(f"{log_prefix} Failed to upload snapshot to HF: {e}")

        logger.info(f"{log_prefix} Backfilling historical snapshots...")
        _backfill_patchtst_snapshots(
            symbols,
            config,
            start_date,
            end_date,
            snapshot_storage,
            storage_backend,
            log_prefix=log_prefix,
        )

    return PatchTSTTrainResponse(
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
        num_input_channels=config.num_input_channels,
        signals_used=["ohlcv"],
    )


@router.post("/patchtst", response_model=PatchTSTTrainResponse)
def train_patchtst(
    background_tasks: BackgroundTasks,
    skip_snapshot: bool = Query(
        False,
        description="Skip saving snapshot (by default saves snapshot for current + all historical years)",
    ),
    storage: PatchTSTModelStorage = Depends(get_patchtst_storage),
    symbols: list[str] = Depends(get_forecaster_training_symbols),
    config: PatchTSTConfig = Depends(get_patchtst_config),
    price_loader: PatchTSTPriceLoader = Depends(get_patchtst_price_loader),
    dataset_builder: PatchTSTDatasetBuilder = Depends(get_patchtst_dataset_builder),
    trainer: PatchTSTTrainer = Depends(get_patchtst_trainer),
) -> PatchTSTTrainResponse | JSONResponse:
    """Train the OHLCV PatchTST model for weekly return prediction.

    Returns 200 with cached result if version already exists (idempotent).
    Returns 202 with job_id if training is started in the background.
    Poll GET /train/status/{job_id} for progress and final result.
    """
    start_date, end_date = resolve_training_window()
    version = patchtst_compute_version(start_date, end_date, symbols, config)
    logger.info(f"[PatchTST] Computed version: {version}")

    if storage.version_exists(version):
        logger.info(f"[PatchTST] Version {version} already exists (idempotent)")
        existing_metadata = storage.read_metadata(version)
        if existing_metadata:
            return PatchTSTTrainResponse(
                version=version,
                data_window_start=existing_metadata["data_window"]["start"],
                data_window_end=existing_metadata["data_window"]["end"],
                metrics=existing_metadata["metrics"],
                promoted=existing_metadata["promoted"],
                prior_version=existing_metadata.get("prior_version"),
                num_input_channels=config.num_input_channels,
                signals_used=["ohlcv"],
            )

    job, is_new = get_or_create_job("patchtst", version)
    if not is_new:
        logger.info(f"[PatchTST] Job {job.job_id} already running, returning 202")
        return JSONResponse(
            status_code=202,
            content=TrainingJobResponse(
                job_id=job.job_id,
                status=job.status,
                message=f"PatchTST training already in progress for {version}",
            ).model_dump(),
        )

    from brain_api.storage.huggingface import PatchTSTHuggingFaceModelStorage

    background_tasks.add_task(
        _run_patchtst_training,
        job_id=job.job_id,
        symbols=symbols,
        storage=storage,
        hf_storage_class=PatchTSTHuggingFaceModelStorage,
        hf_model_repo_getter=get_hf_patchtst_model_repo,
        snapshot_forecaster_type="patchtst",
        skip_snapshot=skip_snapshot,
        config=config,
        price_loader=price_loader,
        dataset_builder=dataset_builder,
        trainer=trainer,
        log_prefix="[PatchTST]",
    )
    logger.info(f"[PatchTST] Background training started: {job.job_id}")

    return JSONResponse(
        status_code=202,
        content=TrainingJobResponse(
            job_id=job.job_id,
            status="pending",
            message=f"PatchTST training started for {version}",
        ).model_dump(),
    )


def _run_patchtst_training(
    *,
    job_id: str,
    symbols: list[str],
    storage: PatchTSTModelStorage,
    hf_storage_class: type,
    hf_model_repo_getter: Callable[[], str | None],
    snapshot_forecaster_type: str,
    skip_snapshot: bool,
    config: PatchTSTConfig,
    price_loader: PatchTSTPriceLoader,
    dataset_builder: PatchTSTDatasetBuilder,
    trainer: PatchTSTTrainer,
    log_prefix: str = "[PatchTST]",
) -> None:
    """Background task that runs the full PatchTST training pipeline."""
    from brain_api.main import shutdown_event

    try:
        response = _train_patchtst_core(
            symbols=symbols,
            storage=storage,
            hf_storage_class=hf_storage_class,
            hf_model_repo_getter=hf_model_repo_getter,
            snapshot_forecaster_type=snapshot_forecaster_type,
            skip_snapshot=skip_snapshot,
            config=config,
            price_loader=price_loader,
            dataset_builder=dataset_builder,
            trainer=trainer,
            log_prefix=log_prefix,
            shutdown_event=shutdown_event,
            job_id=job_id,
        )
        complete_job(job_id, response.model_dump())
        logger.info(f"{log_prefix} Job {job_id} completed successfully")
    except TrainingCancelledError:
        cancel_job(job_id)
        logger.info(f"{log_prefix} Job {job_id} cancelled by shutdown")
    except Exception as e:
        fail_job(job_id, str(e))
        logger.error(f"{log_prefix} Job {job_id} failed: {e}")


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


def _filter_signals_by_cutoff(
    signals: dict[str, pd.DataFrame],
    cutoff_date: date,
) -> dict[str, pd.DataFrame]:
    """Filter signal DataFrames to include only data up to cutoff_date.

    Works with both DatetimeIndex and regular index (will try to convert).

    Args:
        signals: Dict mapping symbol -> DataFrame
        cutoff_date: Include data up to and including this date

    Returns:
        Filtered dict with same structure, excluding symbols with no data after filtering
    """
    cutoff_ts = pd.Timestamp(cutoff_date)
    result = {}

    for symbol, df in signals.items():
        if df.empty:
            continue

        if isinstance(df.index, pd.DatetimeIndex):
            filtered = df[df.index <= cutoff_ts]
        else:
            try:
                idx = pd.to_datetime(df.index)
                mask = idx <= cutoff_ts
                filtered = df[mask]
            except (ValueError, TypeError):
                filtered = df

        if len(filtered) > 0:
            result[symbol] = filtered.copy()

    return result


def _backfill_patchtst_snapshots(
    symbols: list[str],
    config: PatchTSTConfig,
    start_date: date,
    end_date: date,
    snapshot_storage: SnapshotLocalStorage,
    storage_backend: str = "local",
    log_prefix: str = "[PatchTST Backfill]",
) -> None:
    """Backfill PatchTST snapshots for all years that RL walk-forward training needs.

    RL training covering years start_year..end_year needs snapshot-(Y-1)-12-31
    for each year Y.  The earliest snapshot is (start_year-1)-12-31.  To train
    that snapshot we need ``bootstrap_years`` of price history before its cutoff,
    so the price window is extended back to (start_year - 1 - bootstrap_years).

    Optimization: Loads prices ONCE for the extended window and filters
    incrementally by cutoff instead of re-downloading for each snapshot.

    Args:
        symbols: List of stock symbols
        config: PatchTST configuration
        start_date: RL training data start date (from resolve_training_window)
        end_date: Training data end date
        snapshot_storage: Storage instance
        storage_backend: "local" or "hf" - if "hf", uploads to HuggingFace
        log_prefix: Logging prefix string
    """
    backfill_prefix = (
        f"{log_prefix} Backfill" if "Backfill" not in log_prefix else log_prefix
    )
    start_year = start_date.year
    end_year = end_date.year
    bootstrap_years = 4
    check_hf = storage_backend == "hf"

    first_snapshot_year = start_year - 1
    snapshot_data_start = date(first_snapshot_year - bootstrap_years, 1, 1)

    snapshots_needed = []
    for year in range(first_snapshot_year, end_year):
        cutoff_date = date(year, 12, 31)
        if not snapshot_storage.snapshot_exists_anywhere(
            cutoff_date, check_hf=check_hf
        ):
            snapshots_needed.append(cutoff_date)

    if not snapshots_needed:
        logger.info(f"[{backfill_prefix}] All snapshots already exist, nothing to do")
        return

    logger.info(
        f"[{backfill_prefix}] Need to create {len(snapshots_needed)} snapshots: {snapshots_needed}"
    )

    logger.info(
        f"[{backfill_prefix}] Loading prices from {snapshot_data_start} to {end_date}..."
    )
    t0 = time.time()
    prices_full = patchtst_load_prices(symbols, snapshot_data_start, end_date)
    t_prices = time.time() - t0
    logger.info(
        f"[{backfill_prefix}] Loaded prices for {len(prices_full)} symbols in {t_prices:.1f}s"
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
            dataset.X, dataset.y, dataset.feature_scaler, config
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
        )

        snapshot_storage.write_snapshot(
            cutoff_date=cutoff_date,
            model=result.model,
            feature_scaler=result.feature_scaler,
            config=config,
            metadata=metadata,
        )
        logger.info(
            f"[{backfill_prefix}] Saved snapshot for {cutoff_date} in {time.time() - t0:.1f}s"
        )

        if storage_backend == "hf":
            try:
                snapshot_storage.upload_snapshot_to_hf(cutoff_date)
                logger.info(
                    f"[{backfill_prefix}] Uploaded snapshot {cutoff_date} to HuggingFace"
                )
            except Exception as e:
                logger.error(
                    f"[{backfill_prefix}] Failed to upload snapshot to HF: {e}"
                )

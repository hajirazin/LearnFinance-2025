"""LSTM training endpoint."""

import gc
import logging
import time
from datetime import date

import pandas as pd
import torch
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from brain_api.core.config import (
    resolve_training_window,
)
from brain_api.core.lstm import (
    LSTMConfig,
    build_dataset,
    compute_version,
    load_prices_yfinance,
    train_model_pytorch,
)
from brain_api.core.model_buckets import (
    ModelType,
    UnknownBucketError,
    get_bucket,
)
from brain_api.core.training_utils import (
    TrainingCancelledError,
    evaluate_forecaster_artifact_health,
)
from brain_api.storage.forecaster_snapshots import (
    SnapshotLocalStorage,
    create_snapshot_metadata,
)
from brain_api.storage.local import create_metadata
from brain_api.storage.lstm.local import LSTMHalalNewModelStorage
from brain_api.storage.policy import (
    StoragePolicyError,
    get_prior_metadata_for_bucket,
)

from .dependencies import (
    DatasetBuilder,
    PriceLoader,
    Trainer,
    get_config,
    get_dataset_builder,
    get_price_loader,
    get_trainer,
)
from .job_registry import (
    cancel_job,
    complete_job,
    fail_job,
    get_or_create_job,
    update_progress,
)
from .models import LSTMTrainResponse, TrainingJobResponse

router = APIRouter()
logger = logging.getLogger(__name__)

# Universes the US LSTM endpoint accepts. Future India / other-market
# LSTM endpoints get their own router file with their own allowlist;
# the registry (``brain_api.core.model_buckets``) is the source of
# truth for which buckets exist, but each endpoint applies its own
# market policy on top.
_LSTM_US_ALLOWED_UNIVERSES: frozenset[str] = frozenset({"halal_new"})


class LSTMTrainRequest(BaseModel):
    """Body for POST /train/lstm.

    ``universe`` selects the bucket (and therefore the symbol resolver
    + storage path + HF repo). Two parallel workflows can hit this
    endpoint with different ``universe`` values without colliding.
    """

    universe: str = Field(
        default="halal_new",
        description=(
            "Universe to train on. Must be one of the registered LSTM "
            f"buckets exposed by this endpoint: {sorted(_LSTM_US_ALLOWED_UNIVERSES)}."
        ),
    )


@router.post("/lstm", response_model=LSTMTrainResponse)
def train_lstm(
    background_tasks: BackgroundTasks,
    request: LSTMTrainRequest = LSTMTrainRequest(),
    skip_snapshot: bool = Query(
        False,
        description="Skip saving snapshot (by default saves snapshot for current + all historical years)",
    ),
    config: LSTMConfig = Depends(get_config),
    price_loader: PriceLoader = Depends(get_price_loader),
    dataset_builder: DatasetBuilder = Depends(get_dataset_builder),
    trainer: Trainer = Depends(get_trainer),
) -> LSTMTrainResponse | JSONResponse:
    """Train the shared LSTM model for weekly return prediction.

    Returns 200 with cached result if version already exists (idempotent).
    Returns 202 with job_id if training is started in the background.
    Poll GET /train/status/{job_id} for progress and final result.
    """
    if request.universe not in _LSTM_US_ALLOWED_UNIVERSES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown universe {request.universe!r} for /train/lstm. "
                f"Valid options: {sorted(_LSTM_US_ALLOWED_UNIVERSES)}."
            ),
        )
    try:
        bucket = get_bucket(ModelType.LSTM, request.universe)
    except UnknownBucketError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    symbols = bucket.symbols_resolver()
    if bucket.symbol_validator is not None:
        try:
            bucket.symbol_validator(symbols)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    storage: LSTMHalalNewModelStorage = bucket.local_storage_class()

    start_date, end_date = resolve_training_window()
    logger.info(
        f"[LSTM] Starting training for {len(symbols)} symbols "
        f"(bucket={bucket.bucket_name})"
    )
    logger.info(f"[LSTM] Data window: {start_date} to {end_date}")

    version = compute_version(start_date, end_date, symbols, config)
    logger.info(f"[LSTM] Computed version: {version}")

    if storage.version_exists(version):
        logger.info(f"[LSTM] Version {version} already exists (idempotent)")
        existing_metadata = storage.read_metadata(version)
        if existing_metadata:
            return LSTMTrainResponse(
                version=version,
                data_window_start=existing_metadata["data_window"]["start"],
                data_window_end=existing_metadata["data_window"]["end"],
                metrics=existing_metadata["metrics"],
                promoted=existing_metadata["promoted"],
                prior_version=existing_metadata.get("prior_version"),
                # Backward-compat: pre-guardrail metadata files have no
                # ``failure_reasons`` key. Treat missing as empty list.
                failure_reasons=existing_metadata.get("failure_reasons", []),
            )

    job, is_new = get_or_create_job("lstm", version)
    if not is_new:
        logger.info(f"[LSTM] Job {job.job_id} already running, returning 202")
        return JSONResponse(
            status_code=202,
            content=TrainingJobResponse(
                job_id=job.job_id,
                status=job.status,
                message=f"LSTM training already in progress for {version}",
            ).model_dump(),
        )

    background_tasks.add_task(
        _run_lstm_training,
        job_id=job.job_id,
        symbols=symbols,
        config=config,
        storage=storage,
        bucket=bucket,
        price_loader=price_loader,
        dataset_builder=dataset_builder,
        trainer=trainer,
        skip_snapshot=skip_snapshot,
    )
    logger.info(f"[LSTM] Background training started: {job.job_id}")

    return JSONResponse(
        status_code=202,
        content=TrainingJobResponse(
            job_id=job.job_id,
            status="pending",
            message=f"LSTM training started for {version}",
        ).model_dump(),
    )


def _run_lstm_training(
    *,
    job_id: str,
    symbols: list[str],
    config: LSTMConfig,
    storage: LSTMHalalNewModelStorage,
    bucket,
    price_loader: PriceLoader,
    dataset_builder: DatasetBuilder,
    trainer: Trainer,
    skip_snapshot: bool,
) -> None:
    """Background task that runs the full LSTM training pipeline.

    Args:
        bucket: Resolved ``BucketConfig`` from the registry. The
            background task pulls ``hf_repo_getter`` and
            ``hf_storage_class`` from the bucket so HF uploads land on
            the matching repo and snapshots/main-artifacts share one
            source of truth.
    """
    bucket_name = bucket.bucket_name
    hf_repo_getter = bucket.hf_repo_getter
    hf_storage_class = bucket.hf_storage_class

    from brain_api.main import shutdown_event

    try:
        start_date, end_date = resolve_training_window()
        version = compute_version(start_date, end_date, symbols, config)

        update_progress(job_id, {"phase": "loading_prices"})
        logger.info(f"[LSTM] Loading price data for {len(symbols)} symbols...")
        t0 = time.time()
        prices = price_loader(symbols, start_date, end_date)
        t_prices = time.time() - t0
        logger.info(
            f"[LSTM] Loaded prices for {len(prices)}/{len(symbols)} symbols in {t_prices:.1f}s"
        )

        if len(prices) == 0:
            raise ValueError("No price data available for training")

        update_progress(job_id, {"phase": "building_dataset"})
        logger.info("[LSTM] Building dataset...")
        t0 = time.time()
        dataset = dataset_builder(prices, config)
        t_dataset = time.time() - t0
        logger.info(
            f"[LSTM] Dataset built in {t_dataset:.1f}s: {len(dataset.X)} samples"
        )

        if len(dataset.X) == 0:
            raise ValueError("No training samples could be built from price data")

        available_symbols = list(prices.keys())
        X, y, feature_scaler = dataset.X, dataset.y, dataset.feature_scaler
        del dataset, prices
        gc.collect()

        update_progress(job_id, {"phase": "training"})
        logger.info("[LSTM] Starting model training...")
        t0 = time.time()
        result = trainer(X, y, feature_scaler, config, shutdown_event=shutdown_event)
        t_train = time.time() - t0
        logger.info(f"[LSTM] Training complete in {t_train:.1f}s")
        logger.info(
            f"[LSTM] Metrics: train_loss={result.train_loss:.6f}, val_loss={result.val_loss:.6f}, baseline={result.baseline_loss:.6f}"
        )

        update_progress(job_id, {"phase": "promotion_check"})
        hf_model_repo = hf_repo_getter()
        # prior_version is kept purely for audit lineage on metadata.
        # The promotion decision is the artifact health check below;
        # prior metrics are NEVER consulted (they were the source of
        # the universe-drift problem the always-promote refactor fixes).
        try:
            prior_metadata = get_prior_metadata_for_bucket(bucket=bucket)
        except StoragePolicyError as exc:
            logger.warning(
                f"[LSTM] hf_first prior fetch failed for bucket "
                f"{bucket.bucket_name}: {exc}; treating as inaugural"
            )
            prior_metadata = None
        prior_version: str | None = (
            prior_metadata.get("version") if prior_metadata is not None else None
        )

        # Two-write ordering: we must write artifacts to disk first so
        # the file-existence guardrails inside the health check can
        # see them, then re-write metadata.json with the populated
        # ``promoted`` and ``failure_reasons``. Comments next to each
        # write explain which one is the operator-facing copy.
        provisional_metadata = create_metadata(
            version=version,
            data_window_start=start_date.isoformat(),
            data_window_end=end_date.isoformat(),
            symbols=symbols,
            config=config,
            train_loss=result.train_loss,
            val_loss=result.val_loss,
            baseline_loss=result.baseline_loss,
            promoted=False,  # placeholder; real value set below
            prior_version=prior_version,
            failure_reasons=[],  # placeholder
        )

        update_progress(job_id, {"phase": "writing_artifacts"})
        logger.info(f"[LSTM] Writing artifacts for version {version}...")
        version_dir = storage.write_artifacts(
            version=version,
            model=result.model,
            feature_scaler=result.feature_scaler,
            config=config,
            metadata=provisional_metadata,
        )

        health = evaluate_forecaster_artifact_health(
            train_loss=result.train_loss,
            val_loss=result.val_loss,
            baseline_loss=result.baseline_loss,
            artifact_dir=version_dir,
        )
        promoted = health.is_healthy
        logger.info(
            f"[LSTM] Promotion decision: {'PROMOTED' if promoted else 'NOT promoted'}"
            + ("" if promoted else f" (failures: {health.failure_reasons})")
        )

        # Final metadata: rewrite with the real promoted + failure_reasons.
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
            failure_reasons=health.failure_reasons,
        )
        storage.write_artifacts(
            version=version,
            model=result.model,
            feature_scaler=result.feature_scaler,
            config=config,
            metadata=metadata,
        )

        if promoted:
            storage.promote_version(version)
            logger.info(f"[LSTM] Version {version} promoted to current")

        hf_repo = None
        hf_url = None

        # Writes are not policy-gated: whenever the bucket has an HF
        # repo configured we upload, regardless of read policy. Closes
        # audit Bug 6 (uploads were skipped under STORAGE_BACKEND=local
        # even when an HF repo was set).
        if hf_model_repo:
            try:
                hf_storage = hf_storage_class(
                    repo_id=hf_model_repo, local_cache=storage
                )
                # make_current = promoted (no cold-start fallback). An
                # unhealthy inaugural leaves HF main empty and forces
                # the operator to investigate -- per AGENTS.md rule #1,
                # silently shipping bad data is forbidden.
                hf_info = hf_storage.upload_model(
                    version=version,
                    model=result.model,
                    feature_scaler=result.feature_scaler,
                    config=config,
                    metadata=metadata,
                    make_current=promoted,
                )
                hf_repo = hf_info.repo_id
                hf_url = f"https://huggingface.co/{hf_info.repo_id}/tree/{version}"
                logger.info(f"[LSTM] Model uploaded to HuggingFace: {hf_url}")
            except Exception as e:
                logger.error(f"Failed to upload model to HuggingFace: {e}")

        if not skip_snapshot:
            update_progress(job_id, {"phase": "snapshots"})
            snapshot_storage = SnapshotLocalStorage(bucket_name)
            snapshot_hf_repo = snapshot_storage._get_hf_repo()
            check_hf = snapshot_hf_repo is not None

            if not snapshot_storage.snapshot_exists_anywhere(
                end_date, check_hf=check_hf
            ):
                snapshot_metadata = create_snapshot_metadata(
                    forecaster_type=snapshot_storage.forecaster_type,
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
                logger.info(f"[LSTM] Saved snapshot for cutoff {end_date}")

                if snapshot_hf_repo:
                    try:
                        snapshot_storage.upload_snapshot_to_hf(end_date)
                        logger.info(
                            f"[LSTM] Uploaded snapshot {end_date} to HuggingFace"
                        )
                    except Exception as e:
                        logger.error(f"[LSTM] Failed to upload snapshot to HF: {e}")

            logger.info("[LSTM] Backfilling historical snapshots...")
            _backfill_lstm_snapshots(
                symbols, config, start_date, end_date, snapshot_storage
            )

        response = LSTMTrainResponse(
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
            failure_reasons=health.failure_reasons,
            hf_repo=hf_repo,
            hf_url=hf_url,
        )
        complete_job(job_id, response.model_dump())
        logger.info(f"[LSTM] Job {job_id} completed successfully")

    except TrainingCancelledError:
        cancel_job(job_id)
        logger.info(f"[LSTM] Job {job_id} cancelled by shutdown")
    except Exception as e:
        fail_job(job_id, str(e))
        logger.error(f"[LSTM] Job {job_id} failed: {e}")


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
) -> None:
    """Backfill LSTM snapshots for all years that RL walk-forward training needs.

    RL training covering years start_year..end_year needs snapshot-(Y-1)-12-31
    for each year Y.  The earliest snapshot is (start_year-1)-12-31.  To train
    that snapshot we need ``bootstrap_years`` of price history before its cutoff,
    so the price window is extended back to (start_year - 1 - bootstrap_years).

    Optimization: Loads prices ONCE for the extended window and filters
    incrementally by cutoff instead of re-downloading for each snapshot.

    Snapshot uploads are now gated solely on whether the bucket has an
    HF repo configured (``snapshot_storage._get_hf_repo()``) rather
    than on a process-wide policy. This keeps writes consistent
    between ``local_first`` and ``hf_first`` callers and closes audit
    Bug 7.

    Args:
        symbols: List of stock symbols
        config: LSTM configuration
        start_date: RL training data start date (from resolve_training_window)
        end_date: Training data end date
        snapshot_storage: Storage instance
    """
    start_year = start_date.year
    end_year = end_date.year
    bootstrap_years = 4
    snapshot_hf_repo = snapshot_storage._get_hf_repo()
    check_hf = snapshot_hf_repo is not None

    # RL year Y needs snapshot-(Y-1)-12-31.  Create from (start_year-1) onward.
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
        logger.info("[LSTM Backfill] All snapshots already exist, nothing to do")
        return

    logger.info(
        f"[LSTM Backfill] Need to create {len(snapshots_needed)} snapshots: {snapshots_needed}"
    )

    # Load prices ONCE for extended window (covers bootstrap for earliest snapshot)
    logger.info(
        f"[LSTM Backfill] Loading prices from {snapshot_data_start} to {end_date}..."
    )
    t0 = time.time()
    prices_full = load_prices_yfinance(symbols, snapshot_data_start, end_date)
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
            forecaster_type=snapshot_storage.forecaster_type,
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
            f"[LSTM Backfill] Saved snapshot for {cutoff_date} in {time.time() - t0:.1f}s"
        )

        # Upload to HuggingFace whenever the bucket's HF repo env is
        # configured (writes ignore the read policy).
        if snapshot_hf_repo:
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

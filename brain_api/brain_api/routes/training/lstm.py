"""LSTM training endpoint."""

import gc
import logging
import time
from datetime import date

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from brain_api.core.config import (
    resolve_training_window,
)
from brain_api.core.forecaster_snapshot_identity import (
    MissingSnapshotInventory,
    count_missing_snapshots,
)
from brain_api.core.lstm import (
    LSTMConfig,
    compute_version,
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
)
from brain_api.storage.lstm.local import LSTMHalalNewModelStorage
from brain_api.storage.metadata import create_training_metadata
from brain_api.storage.policy import (
    StoragePolicyError,
    build_common_train_response_kwargs,
    get_prior_metadata_for_bucket,
    try_load_existing_train_metadata,
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
from .snapshot_phase import (
    _LSTMMainTrainingArtifacts,
    _run_lstm_snapshot_phase,
)

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

    # HF-aware idempotency skip: under hf_first the helper consults
    # the bucket's HF repo for ``revision=version`` so a wiped local
    # cache does not silently retrain work that already exists on HF.
    # Under local_first behaviour is byte-equivalent to the legacy
    # ``storage.version_exists + read_metadata`` pair.
    existing_metadata = try_load_existing_train_metadata(
        bucket=bucket, version=version, local_storage=storage
    )
    if existing_metadata:
        # Main version is cached. Don't skip outright -- check whether
        # any forecaster snapshot is missing (per AGENTS.md plan: "if
        # main exists, then start checking snapshots, if any are
        # missing, do those"). The scan is policy-aware
        # (``count_missing_snapshots`` consults HF via _resolve_check_hf)
        # so the decision matches the storage backend the operator chose.
        return _handle_lstm_existing_metadata(
            background_tasks=background_tasks,
            bucket=bucket,
            symbols=symbols,
            config=config,
            train_window=(start_date, end_date),
            version=version,
            existing_metadata=existing_metadata,
            skip_snapshot=skip_snapshot,
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


def _handle_lstm_existing_metadata(
    *,
    background_tasks: BackgroundTasks,
    bucket,
    symbols: list[str],
    config: LSTMConfig,
    train_window: tuple[date, date],
    version: str,
    existing_metadata: dict,
    skip_snapshot: bool,
) -> LSTMTrainResponse | JSONResponse:
    """Branch the cached-main response on the snapshot inventory.

    Three outcomes:

    * ``inventory.is_empty`` (or ``skip_snapshot=True``): return 200
      with the cached metadata. Backwards-compatible fast path.
    * Some snapshots missing: schedule the snapshots-only background
      runner under a dedicated job key (``{bucket}_snapshots``) so it
      cannot collide with a real main-training job for the same
      ``version``. Return 202 with the new ``job_id``.
    * ``StoragePolicyError`` raised by ``count_missing_snapshots`` (i.e.
      ``hf_first`` + the snapshot bucket has no HF repo configured):
      surface as 503. Same shape as the inference layer's transient
      config error contract.
    """
    cached_response_kwargs = build_common_train_response_kwargs(
        version, existing_metadata
    )

    if skip_snapshot:
        # Operator explicitly opted out of snapshot bookkeeping. Return
        # cached without scanning -- preserves the legacy contract used
        # by the existing idempotent-rerun tests that pass
        # ``?skip_snapshot=true``.
        logger.info(
            f"[LSTM] Version {version} already exists (idempotent, skip_snapshot=true)"
        )
        return LSTMTrainResponse(**cached_response_kwargs)

    snapshot_storage = SnapshotLocalStorage(bucket.bucket_name)
    try:
        inventory: MissingSnapshotInventory = count_missing_snapshots(
            forecaster_type=bucket.bucket_name,
            train_window=train_window,
            symbols=symbols,
            config_dict=config.to_dict(),
            snapshot_storage=snapshot_storage,
        )
    except StoragePolicyError as exc:
        logger.error(f"[LSTM] Snapshot inventory scan failed for {version}: {exc}")
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if inventory.is_empty:
        logger.info(
            f"[LSTM] Version {version} already exists and all snapshots "
            f"present (idempotent)"
        )
        return LSTMTrainResponse(**cached_response_kwargs)

    snapshots_job_key = f"{bucket.bucket_name}_snapshots"
    job, is_new = get_or_create_job(snapshots_job_key, version)
    if not is_new:
        logger.info(
            f"[LSTM] Snapshots-only job {job.job_id} already in progress for {version}"
        )
        return JSONResponse(
            status_code=202,
            content=TrainingJobResponse(
                job_id=job.job_id,
                status=job.status,
                message=(
                    f"LSTM snapshots-only backfill already in progress for {version}"
                ),
            ).model_dump(),
        )

    background_tasks.add_task(
        _run_lstm_snapshots_only,
        job_id=job.job_id,
        symbols=symbols,
        config=config,
        bucket=bucket,
        train_window=train_window,
        version=version,
        existing_metadata=existing_metadata,
    )
    logger.info(
        f"[LSTM] Snapshots-only backfill started: {job.job_id} "
        f"({inventory.total_missing} cutoff(s) missing)"
    )

    return JSONResponse(
        status_code=202,
        content=TrainingJobResponse(
            job_id=job.job_id,
            status="pending",
            message=(
                f"LSTM snapshots-only backfill started for {version} "
                f"({inventory.total_missing} cutoff(s) missing)"
            ),
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
        provisional_metadata = create_training_metadata(
            model_type=bucket_name,
            version=version,
            data_window_start=start_date.isoformat(),
            data_window_end=end_date.isoformat(),
            symbols=symbols,
            config_dict=config.to_dict(),
            train_loss=result.train_loss,
            val_loss=result.val_loss,
            baseline_loss=result.baseline_loss,
            best_epoch=result.best_epoch,
            stopped_epoch=result.stopped_epoch,
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
        metadata = create_training_metadata(
            model_type=bucket_name,
            version=version,
            data_window_start=start_date.isoformat(),
            data_window_end=end_date.isoformat(),
            symbols=symbols,
            config_dict=config.to_dict(),
            train_loss=result.train_loss,
            val_loss=result.val_loss,
            baseline_loss=result.baseline_loss,
            best_epoch=result.best_epoch,
            stopped_epoch=result.stopped_epoch,
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
            _run_lstm_snapshot_phase(
                train_window=(start_date, end_date),
                symbols=symbols,
                config=config,
                snapshot_storage=snapshot_storage,
                main_artifacts=_LSTMMainTrainingArtifacts(
                    model=result.model,
                    feature_scaler=result.feature_scaler,
                    train_loss=result.train_loss,
                    val_loss=result.val_loss,
                    best_epoch=result.best_epoch,
                    stopped_epoch=result.stopped_epoch,
                    available_symbols=available_symbols,
                ),
            )

        response = LSTMTrainResponse(
            version=version,
            data_window_start=start_date.isoformat(),
            data_window_end=end_date.isoformat(),
            metrics={
                "train_loss": result.train_loss,
                "val_loss": result.val_loss,
                "baseline_loss": result.baseline_loss,
                "best_epoch": result.best_epoch,
                "stopped_epoch": result.stopped_epoch,
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


def _run_lstm_snapshots_only(
    *,
    job_id: str,
    symbols: list[str],
    config: LSTMConfig,
    bucket,
    train_window: tuple[date, date],
    version: str,
    existing_metadata: dict,
) -> None:
    """Background task that runs only the snapshot phase.

    Used by the cached-main path in :func:`_handle_lstm_existing_metadata`
    when at least one snapshot is missing. Skips the entire main
    training pipeline (price load, dataset build, model train,
    artifact write, promotion check, HF upload of main) and goes
    straight to the snapshot phase with ``main_artifacts=None`` so
    the end-of-window snapshot is warned-and-skipped if missing while
    historical year-end snapshots are backfilled by
    :func:`_backfill_lstm_snapshots`.

    On ``StoragePolicyError`` (``hf_first`` + no HF repo) the job is
    marked failed with the policy error message; the route handler
    already mapped that case to 503 synchronously, but a transient
    HF outage between the synchronous scan and the background run is
    still possible.
    """
    try:
        update_progress(job_id, {"phase": "snapshots_only_backfill"})
        snapshot_storage = SnapshotLocalStorage(bucket.bucket_name)
        _run_lstm_snapshot_phase(
            train_window=train_window,
            symbols=symbols,
            config=config,
            snapshot_storage=snapshot_storage,
            main_artifacts=None,
            log_prefix="[LSTM Snapshots-only]",
        )
        response = LSTMTrainResponse(
            **build_common_train_response_kwargs(version, existing_metadata),
        )
        complete_job(job_id, response.model_dump())
        logger.info(f"[LSTM Snapshots-only] Job {job_id} completed successfully")
    except StoragePolicyError as exc:
        fail_job(job_id, str(exc))
        logger.error(f"[LSTM Snapshots-only] Job {job_id} failed (policy): {exc}")
    except Exception as e:
        fail_job(job_id, str(e))
        logger.error(f"[LSTM Snapshots-only] Job {job_id} failed: {e}")

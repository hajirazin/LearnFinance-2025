"""PatchTST training endpoint."""

import gc
import logging
import threading
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
from brain_api.core.model_buckets import (
    BucketConfig,
    ModelType,
    UnknownBucketError,
    get_bucket,
)
from brain_api.core.patchtst import PatchTSTConfig, align_multivariate_data
from brain_api.core.patchtst import (
    compute_version as patchtst_compute_version,
)
from brain_api.core.training_utils import (
    TrainingCancelledError,
    evaluate_forecaster_artifact_health,
)
from brain_api.storage.forecaster_snapshots import (
    SnapshotLocalStorage,
)
from brain_api.storage.metadata import create_training_metadata
from brain_api.storage.patchtst.local import PatchTSTHalalNewModelStorage
from brain_api.storage.policy import (
    StoragePolicyError,
    build_common_train_response_kwargs,
    get_prior_metadata_for_bucket,
    try_load_existing_train_metadata,
)

from .dependencies import (
    PatchTSTDatasetBuilder,
    PatchTSTPriceLoader,
    PatchTSTTrainer,
    get_patchtst_config,
    get_patchtst_dataset_builder,
    get_patchtst_price_loader,
    get_patchtst_trainer,
)
from .job_registry import (
    cancel_job,
    complete_job,
    fail_job,
    get_or_create_job,
    update_progress,
)
from .models import PatchTSTTrainResponse, TrainingJobResponse
from .snapshot_phase import (
    _PatchTSTMainTrainingArtifacts,
    _run_patchtst_snapshot_phase,
)

router = APIRouter()
logger = logging.getLogger(__name__)

# Universes the US PatchTST endpoint accepts. India PatchTST is served
# by ``/train/patchtst/india`` (see patchtst_india.py); each market gets
# its own endpoint per product policy, even though the registry knows
# about both PatchTST buckets.
_PATCHTST_US_ALLOWED_UNIVERSES: frozenset[str] = frozenset({"halal_new"})


class PatchTSTTrainRequest(BaseModel):
    """Body for POST /train/patchtst (US)."""

    universe: str = Field(
        default="halal_new",
        description=(
            "Universe to train on. Must be one of the registered US "
            f"PatchTST buckets: {sorted(_PATCHTST_US_ALLOWED_UNIVERSES)}."
        ),
    )


def _train_patchtst_core(
    symbols: list[str],
    storage: PatchTSTHalalNewModelStorage,
    bucket: BucketConfig,
    skip_snapshot: bool,
    config: PatchTSTConfig,
    price_loader: PatchTSTPriceLoader,
    dataset_builder: PatchTSTDatasetBuilder,
    trainer: PatchTSTTrainer,
    log_prefix: str = "[PatchTST]",
    shutdown_event: threading.Event | None = None,
    job_id: str | None = None,
) -> PatchTSTTrainResponse:
    hf_storage_class = bucket.hf_storage_class
    hf_model_repo_getter = bucket.hf_repo_getter
    snapshot_forecaster_type = bucket.bucket_name
    """Core PatchTST training logic shared by US and India endpoints.

    Handles: version check -> load prices -> align -> build dataset -> train ->
    evaluate promotion -> write artifacts -> HF upload -> snapshot backfill.

    Args:
        symbols: Stock symbols to train on.
        storage: Local model storage instance for the bucket.
        hf_storage_class: HuggingFace storage class for this bucket.
        hf_model_repo_getter: Callable returning HF repo ID.
        snapshot_forecaster_type: Bucket name (e.g.
            ``"patchtst_halal_new"``) used as the snapshot subdirectory
            and HF dispatch key.
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

    # HF-aware idempotency skip: under hf_first the helper consults
    # the bucket's HF repo for ``revision=version`` so a wiped local
    # cache does not silently retrain work that already exists on HF.
    existing_metadata = try_load_existing_train_metadata(
        bucket=bucket, version=version, local_storage=storage
    )
    if existing_metadata:
        logger.info(
            f"{log_prefix} Version {version} already exists (idempotent), returning cached result"
        )
        return PatchTSTTrainResponse(
            **build_common_train_response_kwargs(version, existing_metadata),
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
    anchor_dates = getattr(dataset, "anchor_dates", None)
    del dataset
    gc.collect()

    if job_id:
        update_progress(job_id, {"phase": "training"})
    logger.info(f"{log_prefix} Starting model training...")
    t0 = time.time()
    result = trainer(
        X,
        y,
        feature_scaler,
        config,
        shutdown_event=shutdown_event,
        anchor_dates=anchor_dates,
    )
    t_train = time.time() - t0
    logger.info(f"{log_prefix} Training complete in {t_train:.1f}s")
    logger.info(
        f"{log_prefix} Metrics: train_loss={result.train_loss:.6f}, val_loss={result.val_loss:.6f}, baseline={result.baseline_loss:.6f}"
    )

    hf_model_repo = hf_model_repo_getter()
    # prior_version is kept purely for audit lineage on metadata. The
    # promotion decision is the artifact health check below; prior
    # metrics are NEVER consulted (universe-drift made them an
    # apples-to-oranges baseline).
    try:
        prior_metadata = get_prior_metadata_for_bucket(bucket=bucket)
    except StoragePolicyError as exc:
        logger.warning(
            f"{log_prefix} hf_first prior fetch failed for bucket "
            f"{bucket.bucket_name}: {exc}; treating as inaugural"
        )
        prior_metadata = None
    prior_version: str | None = (
        prior_metadata.get("version") if prior_metadata is not None else None
    )

    # Two-write ordering: write artifacts first so the file-existence
    # guardrails inside the health check can observe them, then
    # re-write metadata.json with the populated promoted +
    # failure_reasons.
    provisional_metadata = create_training_metadata(
        model_type=bucket.bucket_name,
        version=version,
        data_window_start=start_date.isoformat(),
        data_window_end=end_date.isoformat(),
        symbols=symbols,
        config_dict=config.to_dict(),
        train_loss=result.train_loss,
        val_loss=result.val_loss,
        baseline_loss=result.baseline_loss,
        promoted=False,  # placeholder
        prior_version=prior_version,
        failure_reasons=[],  # placeholder
    )

    logger.info(f"{log_prefix} Writing artifacts for version {version}...")
    version_dir = storage.write_artifacts(
        version=version,
        model=result.model,
        feature_scaler=result.feature_scaler,
        config=config,
        metadata=provisional_metadata,
    )
    logger.info(f"{log_prefix} Artifacts written successfully")

    health = evaluate_forecaster_artifact_health(
        train_loss=result.train_loss,
        val_loss=result.val_loss,
        baseline_loss=result.baseline_loss,
        artifact_dir=version_dir,
    )
    promoted = health.is_healthy
    logger.info(
        f"{log_prefix} Promotion decision: {'PROMOTED' if promoted else 'NOT promoted'}"
        + ("" if promoted else f" (failures: {health.failure_reasons})")
    )

    # Final metadata write with the real promoted + failure_reasons.
    metadata = create_training_metadata(
        model_type=bucket.bucket_name,
        version=version,
        data_window_start=start_date.isoformat(),
        data_window_end=end_date.isoformat(),
        symbols=symbols,
        config_dict=config.to_dict(),
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
        logger.info(f"{log_prefix} Version {version} promoted to current")

    hf_repo = None
    hf_url = None

    # Writes ignore the read policy: upload whenever the bucket has an
    # HF repo configured. Closes audit Bug 6.
    if hf_model_repo:
        try:
            hf_storage = hf_storage_class(repo_id=hf_model_repo, local_cache=storage)
            # make_current = promoted (no cold-start fallback). An
            # unhealthy inaugural leaves HF main empty and forces the
            # operator to investigate -- per AGENTS.md rule #1.
            logger.info(
                f"{log_prefix} HF upload: promoted={promoted} "
                f"-> make_current={promoted}"
            )

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
            logger.info(f"{log_prefix} Model uploaded to HuggingFace: {hf_url}")
        except Exception as e:
            logger.error(f"{log_prefix} Failed to upload model to HuggingFace: {e}")

    if not skip_snapshot:
        snapshot_storage = SnapshotLocalStorage(snapshot_forecaster_type)
        _run_patchtst_snapshot_phase(
            train_window=(start_date, end_date),
            symbols=symbols,
            config=config,
            snapshot_storage=snapshot_storage,
            main_artifacts=_PatchTSTMainTrainingArtifacts(
                model=result.model,
                feature_scaler=result.feature_scaler,
                train_loss=result.train_loss,
                val_loss=result.val_loss,
                available_symbols=available_symbols,
            ),
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
        failure_reasons=health.failure_reasons,
        hf_repo=hf_repo,
        hf_url=hf_url,
        num_input_channels=config.num_input_channels,
        signals_used=["ohlcv"],
    )


@router.post("/patchtst", response_model=PatchTSTTrainResponse)
def train_patchtst(
    background_tasks: BackgroundTasks,
    request: PatchTSTTrainRequest = PatchTSTTrainRequest(),
    skip_snapshot: bool = Query(
        False,
        description="Skip saving snapshot (by default saves snapshot for current + all historical years)",
    ),
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
    if request.universe not in _PATCHTST_US_ALLOWED_UNIVERSES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown universe {request.universe!r} for /train/patchtst. "
                f"Valid options: {sorted(_PATCHTST_US_ALLOWED_UNIVERSES)}."
            ),
        )
    try:
        bucket = get_bucket(ModelType.PATCHTST, request.universe)
    except UnknownBucketError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    symbols = bucket.symbols_resolver()
    if bucket.symbol_validator is not None:
        try:
            bucket.symbol_validator(symbols)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    storage: PatchTSTHalalNewModelStorage = bucket.local_storage_class()

    start_date, end_date = resolve_training_window()
    version = patchtst_compute_version(start_date, end_date, symbols, config)
    logger.info(f"[PatchTST] Computed version: {version} (bucket={bucket.bucket_name})")

    # HF-aware idempotency skip (see ``_train_patchtst_core`` for the
    # broader rationale). Duplicated here because the route handler
    # short-circuits BEFORE enqueueing the background task -- so the
    # cached response is returned synchronously rather than via the
    # 202 -> poll cycle.
    existing_metadata = try_load_existing_train_metadata(
        bucket=bucket, version=version, local_storage=storage
    )
    if existing_metadata:
        return handle_patchtst_existing_metadata(
            background_tasks=background_tasks,
            bucket=bucket,
            symbols=symbols,
            config=config,
            train_window=(start_date, end_date),
            version=version,
            existing_metadata=existing_metadata,
            skip_snapshot=skip_snapshot,
            log_prefix="[PatchTST]",
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

    background_tasks.add_task(
        _run_patchtst_training,
        job_id=job.job_id,
        symbols=symbols,
        storage=storage,
        bucket=bucket,
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


def handle_patchtst_existing_metadata(
    *,
    background_tasks: BackgroundTasks,
    bucket: BucketConfig,
    symbols: list[str],
    config: PatchTSTConfig,
    train_window: tuple[date, date],
    version: str,
    existing_metadata: dict,
    skip_snapshot: bool,
    log_prefix: str,
) -> PatchTSTTrainResponse | JSONResponse:
    """Branch the cached-main response on the snapshot inventory.

    Shared by US (``/train/patchtst``) and India (``/train/patchtst/india``)
    routes. India imports this directly so the bucket-aware path lives
    in one place.

    Three outcomes:

    * ``inventory.is_empty`` (or ``skip_snapshot=True``): return 200
      cached. Backwards-compatible fast path.
    * Some snapshots missing: schedule snapshots-only background runner
      under a dedicated ``{bucket}_snapshots`` key. Return 202.
    * ``StoragePolicyError`` from ``count_missing_snapshots`` (i.e.
      ``hf_first`` + the snapshot bucket has no HF repo configured):
      surface as 503.
    """
    cached_response_kwargs = build_common_train_response_kwargs(
        version, existing_metadata
    )

    if skip_snapshot:
        logger.info(
            f"{log_prefix} Version {version} already exists (idempotent, "
            f"skip_snapshot=true)"
        )
        return PatchTSTTrainResponse(
            **cached_response_kwargs,
            num_input_channels=config.num_input_channels,
            signals_used=["ohlcv"],
        )

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
        logger.error(
            f"{log_prefix} Snapshot inventory scan failed for {version}: {exc}"
        )
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if inventory.is_empty:
        logger.info(
            f"{log_prefix} Version {version} already exists and all "
            f"snapshots present (idempotent)"
        )
        return PatchTSTTrainResponse(
            **cached_response_kwargs,
            num_input_channels=config.num_input_channels,
            signals_used=["ohlcv"],
        )

    snapshots_job_key = f"{bucket.bucket_name}_snapshots"
    job, is_new = get_or_create_job(snapshots_job_key, version)
    if not is_new:
        logger.info(
            f"{log_prefix} Snapshots-only job {job.job_id} already in "
            f"progress for {version}"
        )
        return JSONResponse(
            status_code=202,
            content=TrainingJobResponse(
                job_id=job.job_id,
                status=job.status,
                message=(
                    f"PatchTST snapshots-only backfill already in progress "
                    f"for {version}"
                ),
            ).model_dump(),
        )

    background_tasks.add_task(
        _run_patchtst_snapshots_only,
        job_id=job.job_id,
        symbols=symbols,
        config=config,
        bucket=bucket,
        train_window=train_window,
        version=version,
        existing_metadata=existing_metadata,
        log_prefix=f"{log_prefix} Snapshots-only",
    )
    logger.info(
        f"{log_prefix} Snapshots-only backfill started: {job.job_id} "
        f"({inventory.total_missing} cutoff(s) missing)"
    )

    return JSONResponse(
        status_code=202,
        content=TrainingJobResponse(
            job_id=job.job_id,
            status="pending",
            message=(
                f"PatchTST snapshots-only backfill started for {version} "
                f"({inventory.total_missing} cutoff(s) missing)"
            ),
        ).model_dump(),
    )


def _run_patchtst_training(
    *,
    job_id: str,
    symbols: list[str],
    storage: PatchTSTHalalNewModelStorage,
    bucket: BucketConfig,
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
            bucket=bucket,
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


def _run_patchtst_snapshots_only(
    *,
    job_id: str,
    symbols: list[str],
    config: PatchTSTConfig,
    bucket: BucketConfig,
    train_window: tuple[date, date],
    version: str,
    existing_metadata: dict,
    log_prefix: str = "[PatchTST Snapshots-only]",
) -> None:
    """Background task that runs only the snapshot phase.

    Mirror of :func:`brain_api.routes.training.lstm._run_lstm_snapshots_only`.

    Used by the cached-main path in :func:`handle_patchtst_existing_metadata`
    when at least one snapshot is missing. Skips the entire main
    training pipeline and goes straight to the snapshot phase with
    ``main_artifacts=None`` so the end-of-window snapshot is
    warned-and-skipped if missing while historical year-end snapshots
    are backfilled.

    On ``StoragePolicyError`` (``hf_first`` + no HF repo) the job is
    marked failed; the route handler already mapped that case to 503
    synchronously, but a transient HF outage between the synchronous
    scan and the background run is still possible.
    """
    try:
        update_progress(job_id, {"phase": "snapshots_only_backfill"})
        snapshot_storage = SnapshotLocalStorage(bucket.bucket_name)
        _run_patchtst_snapshot_phase(
            train_window=train_window,
            symbols=symbols,
            config=config,
            snapshot_storage=snapshot_storage,
            main_artifacts=None,
            log_prefix=log_prefix,
        )
        response = PatchTSTTrainResponse(
            **build_common_train_response_kwargs(version, existing_metadata),
            num_input_channels=config.num_input_channels,
            signals_used=["ohlcv"],
        )
        complete_job(job_id, response.model_dump())
        logger.info(f"{log_prefix} Job {job_id} completed successfully")
    except StoragePolicyError as exc:
        fail_job(job_id, str(exc))
        logger.error(f"{log_prefix} Job {job_id} failed (policy): {exc}")
    except Exception as e:
        fail_job(job_id, str(e))
        logger.error(f"{log_prefix} Job {job_id} failed: {e}")

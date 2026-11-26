"""Cached-main and background-job orchestration for PatchTST training."""

from __future__ import annotations

import logging
from datetime import date

from fastapi import BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse

from brain_api.core.forecaster_snapshot_identity import (
    MissingSnapshotInventory,
    count_missing_snapshots,
)
from brain_api.core.model_buckets import BucketConfig
from brain_api.core.patchtst import PatchTSTConfig
from brain_api.core.patchtst.training_evidence import validate_label_evidence
from brain_api.core.training_utils import TrainingCancelledError
from brain_api.storage.forecaster_snapshots import SnapshotLocalStorage
from brain_api.storage.patchtst.local import PatchTSTHalalNewModelStorage
from brain_api.storage.policy import (
    StoragePolicyError,
    build_common_train_response_kwargs,
)

from .dependencies import (
    PatchTSTDatasetBuilder,
    PatchTSTPriceLoader,
    PatchTSTTrainer,
)
from .job_registry import cancel_job, complete_job, fail_job, get_or_create_job
from .models import PatchTSTTrainResponse, TrainingJobResponse

logger = logging.getLogger(__name__)


def handle_patchtst_existing_metadata(
    *,
    background_tasks: BackgroundTasks,
    bucket: BucketConfig,
    symbols: list[str],
    config: PatchTSTConfig,
    price_loader: PatchTSTPriceLoader,
    train_window: tuple[date, date],
    version: str,
    existing_metadata: dict,
    skip_snapshot: bool,
    log_prefix: str,
) -> PatchTSTTrainResponse | JSONResponse:
    """Branch a cached PatchTST main result on its snapshot inventory."""
    from .patchtst import _run_patchtst_snapshots_only

    cached_response_kwargs = build_common_train_response_kwargs(
        version, existing_metadata
    )
    if config.weekly_regression:
        try:
            validate_label_evidence(
                existing_metadata.get("training_evidence", {}),
                config.market,
                date.fromisoformat(existing_metadata["data_window"]["end"]),
                len(symbols),
            )
        except ValueError as exc:
            raise HTTPException(503, str(exc)) from exc
    local = bucket.local_storage_class()
    needs_upload = (
        config.weekly_regression
        and local.version_exists(version)
        and bool(bucket.hf_repo_getter())
        and (
            existing_metadata.get("publication_pending")
            or existing_metadata.get("hf_repo") != bucket.hf_repo_getter()
        )
    )

    if skip_snapshot and not needs_upload:
        logger.info(
            f"{log_prefix} Version {version} already exists (idempotent, "
            "skip_snapshot=true)"
        )
        return PatchTSTTrainResponse(
            **cached_response_kwargs,
            num_input_channels=config.num_input_channels,
            signals_used=config.feature_names
            if config.weekly_regression
            else ["ohlcv"],
        )

    snapshot_storage = SnapshotLocalStorage(bucket.bucket_name)
    if config.weekly_regression and bucket.universe == "halal_new":
        from datetime import timedelta

        from .weekly_patchtst_snapshots import EARLIEST_US_SNAPSHOT

        first = existing_metadata.get("training_evidence", {}).get(
            "first_snapshot_cutoff"
        )
        cutoff = date.fromisoformat(first) if first else EARLIEST_US_SNAPSHOT
        train_window = (
            max(train_window[0], cutoff + timedelta(days=1)),
            train_window[1],
        )
    try:
        inventory: MissingSnapshotInventory = (
            MissingSnapshotInventory(())
            if skip_snapshot
            else count_missing_snapshots(
                forecaster_type=bucket.bucket_name,
                train_window=train_window,
                config_dict=config.to_dict(),
                snapshot_storage=snapshot_storage,
            )
        )
    except StoragePolicyError as exc:
        logger.error(
            f"{log_prefix} Snapshot inventory scan failed for {version}: {exc}"
        )
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if (
        inventory.is_empty
        and not needs_upload
        and not existing_metadata.get("snapshots_pending")
    ):
        logger.info(
            f"{log_prefix} Version {version} already exists and all "
            "snapshots present (idempotent)"
        )
        return PatchTSTTrainResponse(
            **cached_response_kwargs,
            num_input_channels=config.num_input_channels,
            signals_used=config.feature_names
            if config.weekly_regression
            else ["ohlcv"],
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
                    "PatchTST snapshots-only backfill already in progress "
                    f"for {version}"
                ),
            ).model_dump(),
        )

    background_tasks.add_task(
        _run_patchtst_snapshots_only,
        job_id=job.job_id,
        symbols=symbols,
        config=config,
        price_loader=price_loader,
        bucket=bucket,
        train_window=train_window,
        version=version,
        existing_metadata=existing_metadata,
        skip_snapshot=skip_snapshot,
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
    """Run the full PatchTST training pipeline as a background job."""
    from brain_api.main import shutdown_event

    from .patchtst import _train_patchtst_core

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
    except Exception as exc:
        fail_job(job_id, str(exc))
        logger.error(f"{log_prefix} Job {job_id} failed: {exc}")

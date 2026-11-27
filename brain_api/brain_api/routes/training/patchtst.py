"""PatchTST training endpoint."""

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from brain_api.core.config import (
    resolve_training_window,
)
from brain_api.core.model_buckets import (
    ModelType,
    UnknownBucketError,
    get_bucket,
)
from brain_api.core.patchtst import PatchTSTConfig
from brain_api.core.patchtst import (
    compute_version as patchtst_compute_version,
)
from brain_api.core.patchtst.training_evidence import can_reuse_weekly_fit
from brain_api.storage.patchtst.local import PatchTSTHalalNewModelStorage
from brain_api.storage.policy import (
    try_load_existing_train_metadata,
)

from .dependencies import (
    PatchTSTPriceLoader,
    WeeklyPatchTSTTrainer,
    get_patchtst_config,
    get_patchtst_price_loader,
    get_weekly_patchtst_trainer,
)
from .job_registry import (
    get_or_create_job,
)
from .models import PatchTSTTrainResponse, TrainingJobResponse
from .patchtst_jobs import (
    _run_patchtst_training,
    handle_patchtst_existing_metadata,
)
from .patchtst_recovery import (
    _run_patchtst_snapshots_only as _run_patchtst_snapshots_only,
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
    trainer: WeeklyPatchTSTTrainer = Depends(get_weekly_patchtst_trainer),
) -> PatchTSTTrainResponse | JSONResponse:
    """Train the US seven-feature PatchTST weekly return regressor.

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

    # Consult the bucket's configured storage before queueing a fit; completed
    # artifacts can return synchronously or resume missing publication work.
    existing_metadata = try_load_existing_train_metadata(
        bucket=bucket, version=version, local_storage=storage
    )
    if existing_metadata and can_reuse_weekly_fit(
        existing_metadata, config, len(symbols)
    ):
        return handle_patchtst_existing_metadata(
            background_tasks=background_tasks,
            bucket=bucket,
            symbols=symbols,
            config=config,
            price_loader=price_loader,
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
        training_window=(start_date, end_date),
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

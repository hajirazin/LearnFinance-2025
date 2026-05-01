"""India PatchTST training endpoint.

Trains a PatchTST model on the ``nifty_shariah_500`` universe (~210
``.NS``-suffixed Indian Shariah constituents). Artifacts live in their
own bucket (``data/models/patchtst_nifty_shariah_500/``) with an
independent ``current`` pointer; promoting India PatchTST MUST NOT
touch the US ``patchtst_halal_new`` pointer.

Per AGENTS.md, the broad universe is the right tier for forecaster
training. The sticky-15 ``halal_india`` slate is for future India SAC
training (separate endpoint), not for PatchTST itself.
"""

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from brain_api.core.config import resolve_training_window
from brain_api.core.model_buckets import (
    ModelType,
    UnknownBucketError,
    get_bucket,
)
from brain_api.core.patchtst import PatchTSTConfig
from brain_api.core.patchtst import compute_version as patchtst_compute_version
from brain_api.storage.patchtst.local import PatchTSTNiftyShariah500ModelStorage

from .dependencies import (
    PatchTSTDatasetBuilder,
    PatchTSTPriceLoader,
    PatchTSTTrainer,
    get_patchtst_config,
    get_patchtst_dataset_builder,
    get_patchtst_price_loader,
    get_patchtst_trainer,
)
from .job_registry import get_or_create_job
from .models import PatchTSTTrainResponse, TrainingJobResponse
from .patchtst import _run_patchtst_training

router = APIRouter()
logger = logging.getLogger(__name__)

# India PatchTST is single-bucket today; an A/B universe (e.g. a
# liquid-only India slate) would be added by registering a new bucket
# and extending this allowlist.
_PATCHTST_INDIA_ALLOWED_UNIVERSES: frozenset[str] = frozenset({"nifty_shariah_500"})


class PatchTSTIndiaTrainRequest(BaseModel):
    """Body for POST /train/patchtst/india."""

    universe: str = Field(
        default="nifty_shariah_500",
        description=(
            "India PatchTST universe. Must be one of: "
            f"{sorted(_PATCHTST_INDIA_ALLOWED_UNIVERSES)}."
        ),
    )


@router.post("/patchtst/india", response_model=PatchTSTTrainResponse)
def train_patchtst_india(
    background_tasks: BackgroundTasks,
    request: PatchTSTIndiaTrainRequest = PatchTSTIndiaTrainRequest(),
    skip_snapshot: bool = Query(
        False,
        description="Skip saving snapshot (by default saves snapshot for current + all historical years)",
    ),
    config: PatchTSTConfig = Depends(get_patchtst_config),
    price_loader: PatchTSTPriceLoader = Depends(get_patchtst_price_loader),
    dataset_builder: PatchTSTDatasetBuilder = Depends(get_patchtst_dataset_builder),
    trainer: PatchTSTTrainer = Depends(get_patchtst_trainer),
) -> PatchTSTTrainResponse | JSONResponse:
    """Train the India OHLCV PatchTST model for weekly return prediction.

    Returns 200 with cached result if version already exists (idempotent).
    Returns 202 with job_id if training is started in the background.
    Poll GET /train/status/{job_id} for progress and final result.
    """
    if request.universe not in _PATCHTST_INDIA_ALLOWED_UNIVERSES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown universe {request.universe!r} for /train/patchtst/india. "
                f"Valid options: {sorted(_PATCHTST_INDIA_ALLOWED_UNIVERSES)}."
            ),
        )
    try:
        bucket = get_bucket(ModelType.PATCHTST, request.universe)
    except UnknownBucketError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    symbols = bucket.symbols_resolver()
    # India bucket has the .NS-suffix validator wired in the registry;
    # raise as 422 if the upstream universe builder ever returns a
    # mistakenly-stripped symbol (per AGENTS.md no silent fallback).
    if bucket.symbol_validator is not None:
        try:
            bucket.symbol_validator(symbols)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    storage: PatchTSTNiftyShariah500ModelStorage = bucket.local_storage_class()

    start_date, end_date = resolve_training_window()
    version = patchtst_compute_version(start_date, end_date, symbols, config)
    logger.info(
        f"[PatchTST India] Computed version: {version} (bucket={bucket.bucket_name})"
    )

    if storage.version_exists(version):
        logger.info(f"[PatchTST India] Version {version} already exists (idempotent)")
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

    job, is_new = get_or_create_job("patchtst_india", version)
    if not is_new:
        logger.info(f"[PatchTST India] Job {job.job_id} already running, returning 202")
        return JSONResponse(
            status_code=202,
            content=TrainingJobResponse(
                job_id=job.job_id,
                status=job.status,
                message=f"PatchTST India training already in progress for {version}",
            ).model_dump(),
        )

    background_tasks.add_task(
        _run_patchtst_training,
        job_id=job.job_id,
        symbols=symbols,
        storage=storage,
        hf_storage_class=bucket.hf_storage_class,
        hf_model_repo_getter=bucket.hf_repo_getter,
        snapshot_forecaster_type=bucket.bucket_name,
        skip_snapshot=skip_snapshot,
        config=config,
        price_loader=price_loader,
        dataset_builder=dataset_builder,
        trainer=trainer,
        log_prefix="[PatchTST India]",
    )
    logger.info(f"[PatchTST India] Background training started: {job.job_id}")

    return JSONResponse(
        status_code=202,
        content=TrainingJobResponse(
            job_id=job.job_id,
            status="pending",
            message=f"PatchTST India training started for {version}",
        ).model_dump(),
    )

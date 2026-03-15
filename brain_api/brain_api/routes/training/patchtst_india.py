"""India PatchTST training endpoint.

Trains a PatchTST model on NiftyShariah500 (.NS-suffixed) symbols,
storing artifacts separately under data/models/patchtst_india/.
Reuses the shared _train_patchtst_core() so no training logic is duplicated.
"""

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, Query
from fastapi.responses import JSONResponse

from brain_api.core.config import (
    get_hf_patchtst_india_model_repo,
    resolve_training_window,
)
from brain_api.core.patchtst import PatchTSTConfig
from brain_api.core.patchtst import compute_version as patchtst_compute_version
from brain_api.storage.patchtst.huggingface import (
    PatchTSTIndiaHuggingFaceModelStorage,
)
from brain_api.storage.patchtst.local import PatchTSTIndiaModelStorage
from brain_api.universe.nifty_shariah_500 import get_nifty_shariah_500_symbols

from .dependencies import (
    PatchTSTDatasetBuilder,
    PatchTSTPriceLoader,
    PatchTSTTrainer,
    get_patchtst_config,
    get_patchtst_dataset_builder,
    get_patchtst_india_storage,
    get_patchtst_price_loader,
    get_patchtst_trainer,
)
from .job_registry import get_or_create_job
from .models import PatchTSTTrainResponse, TrainingJobResponse
from .patchtst import _run_patchtst_training

router = APIRouter()
logger = logging.getLogger(__name__)


def _get_india_symbols() -> list[str]:
    """Get NiftyShariah500 symbols for India PatchTST training."""
    return get_nifty_shariah_500_symbols()


@router.post("/patchtst/india", response_model=PatchTSTTrainResponse)
def train_patchtst_india(
    background_tasks: BackgroundTasks,
    skip_snapshot: bool = Query(
        False,
        description="Skip saving snapshot (by default saves snapshot for current + all historical years)",
    ),
    storage: PatchTSTIndiaModelStorage = Depends(get_patchtst_india_storage),
    symbols: list[str] = Depends(_get_india_symbols),
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
    start_date, end_date = resolve_training_window()
    version = patchtst_compute_version(start_date, end_date, symbols, config)
    logger.info(f"[PatchTST India] Computed version: {version}")

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
        hf_storage_class=PatchTSTIndiaHuggingFaceModelStorage,
        hf_model_repo_getter=get_hf_patchtst_india_model_repo,
        snapshot_forecaster_type="patchtst_india",
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

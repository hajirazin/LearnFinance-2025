"""India PatchTST training endpoint.

Trains a PatchTST model on NiftyShariah500 (.NS-suffixed) symbols,
storing artifacts separately under data/models/patchtst_india/.
Reuses the shared _train_patchtst_core() so no training logic is duplicated.
"""

import logging

from fastapi import APIRouter, Depends, Query

from brain_api.core.config import get_hf_patchtst_india_model_repo
from brain_api.core.patchtst import PatchTSTConfig
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
from .models import PatchTSTTrainResponse
from .patchtst import _train_patchtst_core

router = APIRouter()
logger = logging.getLogger(__name__)


def _get_india_symbols() -> list[str]:
    """Get NiftyShariah500 symbols for India PatchTST training."""
    return get_nifty_shariah_500_symbols()


@router.post("/patchtst/india", response_model=PatchTSTTrainResponse)
def train_patchtst_india(
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
) -> PatchTSTTrainResponse:
    """Train the India OHLCV PatchTST model for weekly return prediction.

    Same architecture as the US PatchTST model, but trained on NiftyShariah500
    stocks (~210 NSE India symbols with .NS suffix). Artifacts are stored
    under data/models/patchtst_india/ with an independent version pointer.

    By default, also saves snapshots for all historical years.
    Use skip_snapshot=true to disable.

    Returns:
        Training result including version, metrics, and promotion status.
    """
    return _train_patchtst_core(
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

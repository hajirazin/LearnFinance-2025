"""PatchTST inference endpoint."""

import logging
import time

from fastapi import APIRouter, Depends, HTTPException

from brain_api.core.inference_utils import compute_week_from_cutoff
from brain_api.core.patchtst.inference import run_batch_inference
from brain_api.storage.local import PatchTSTModelStorage

from .dependencies import get_patchtst_as_of_date, get_patchtst_storage
from .models import PatchTSTInferenceRequest, PatchTSTInferenceResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/patchtst", response_model=PatchTSTInferenceResponse)
def infer_patchtst(
    request: PatchTSTInferenceRequest,
    storage: PatchTSTModelStorage = Depends(get_patchtst_storage),
) -> PatchTSTInferenceResponse:
    """Predict weekly returns using the current PatchTST model's symbols.

    Symbols are resolved from the current model's training metadata,
    ensuring inference always runs on exactly the symbols the model was trained on.

    Returns:
        PatchTSTInferenceResponse with per-symbol predictions and metadata

    Raises:
        HTTPException 400: if no current model version is available
        HTTPException 503: if model artifacts cannot be loaded
    """
    t_start = time.time()

    # Resolve symbols from current model metadata
    version = storage.read_current_version()
    if not version:
        raise HTTPException(400, "No current PatchTST model version available")
    metadata = storage.read_metadata(version)
    if not metadata or "symbols" not in metadata:
        raise HTTPException(400, f"PatchTST model {version} has no symbols in metadata")
    symbols: list[str] = metadata["symbols"]

    logger.info(
        f"[PatchTST] Starting inference for {len(symbols)} symbols (model {version})"
    )

    cutoff_date = get_patchtst_as_of_date(request)
    logger.info(f"[PatchTST] Cutoff date: {cutoff_date}")

    week_boundaries = compute_week_from_cutoff(cutoff_date)

    try:
        batch_result = run_batch_inference(symbols, cutoff_date, storage)
    except ValueError as e:
        raise HTTPException(503, str(e)) from e

    predictions = batch_result.predictions
    valid_predictions = [
        p for p in predictions if p.predicted_weekly_return_pct is not None
    ]

    t_total = time.time() - t_start
    logger.info(
        f"[PatchTST] Request complete: {len(valid_predictions)}/{len(symbols)} predictions in {t_total:.2f}s"
    )
    if valid_predictions:
        top = valid_predictions[0]
        bottom = valid_predictions[-1]
        logger.info(
            f"[PatchTST] Top: {top.symbol} ({top.predicted_weekly_return_pct:+.2f}%), "
            f"Bottom: {bottom.symbol} ({bottom.predicted_weekly_return_pct:+.2f}%)"
        )

    return PatchTSTInferenceResponse(
        predictions=predictions,
        model_version=batch_result.model_version,
        as_of_date=cutoff_date.isoformat(),
        target_week_start=week_boundaries.target_week_start.isoformat(),
        target_week_end=week_boundaries.target_week_end.isoformat(),
        signals_used=["ohlcv"],
    )

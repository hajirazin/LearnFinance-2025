"""PatchTST inference endpoints (US + India + score-batch).

Three routes:

* ``POST /inference/patchtst`` -- US model, full per-symbol response.
* ``POST /inference/patchtst/india`` -- India model, same response shape.
* ``POST /inference/patchtst/score-batch`` -- wraps batch inference with
  the rank-band score validation policy (non-finite rejection +
  ``min_predictions`` floor) so callers (Temporal Alpha-HRP activities)
  can be pure HTTP wrappers. ``market`` selects the storage backend.

The forward-pass math (``run_batch_inference``) is one implementation
across markets -- only the trained weights/scalers differ. Validation
math (``validate_and_collect_finite_scores``) is also one implementation
in core.
"""

import logging
import time

from fastapi import APIRouter, Depends, HTTPException

from brain_api.core.inference_utils import compute_week_from_cutoff
from brain_api.core.patchtst import validate_and_collect_finite_scores
from brain_api.core.patchtst.inference import run_batch_inference
from brain_api.storage.local import PatchTSTIndiaModelStorage, PatchTSTModelStorage

from .dependencies import (
    get_patchtst_as_of_date,
    get_patchtst_india_storage,
    get_patchtst_storage,
)
from .models import (
    PatchTSTInferenceRequest,
    PatchTSTInferenceResponse,
    PatchTSTScoreBatchRequest,
    PatchTSTScoreBatchResponse,
)

router = APIRouter()
logger = logging.getLogger(__name__)


def _run_patchtst_inference(
    request: PatchTSTInferenceRequest,
    storage: PatchTSTModelStorage,
    log_prefix: str,
) -> PatchTSTInferenceResponse:
    """Run a single-market PatchTST inference and shape the response.

    Shared between ``/patchtst`` (US) and ``/patchtst/india`` so the
    forward-pass + response-assembly math has one implementation -- only
    the storage backend (trained weights/scalers) differs per market.
    """
    t_start = time.time()

    version = storage.read_current_version()
    if not version:
        raise HTTPException(400, "No current PatchTST model version available")

    if request.symbols is not None:
        symbols = list(request.symbols)
        logger.info(
            f"{log_prefix} Using {len(symbols)} requested symbols (model {version})"
        )
    else:
        metadata = storage.read_metadata(version)
        if not metadata or "symbols" not in metadata:
            raise HTTPException(
                400, f"PatchTST model {version} has no symbols in metadata"
            )
        symbols = metadata["symbols"]
        logger.info(
            f"{log_prefix} Starting inference for {len(symbols)} symbols from metadata "
            f"(model {version})"
        )

    cutoff_date = get_patchtst_as_of_date(request)
    logger.info(f"{log_prefix} Cutoff date: {cutoff_date}")

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
        f"{log_prefix} Request complete: {len(valid_predictions)}/{len(symbols)} "
        f"predictions in {t_total:.2f}s"
    )
    if valid_predictions:
        top = valid_predictions[0]
        bottom = valid_predictions[-1]
        logger.info(
            f"{log_prefix} Top: {top.symbol} ({top.predicted_weekly_return_pct:+.2f}%), "
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


@router.post("/patchtst", response_model=PatchTSTInferenceResponse)
def infer_patchtst(
    request: PatchTSTInferenceRequest,
    storage: PatchTSTModelStorage = Depends(get_patchtst_storage),
) -> PatchTSTInferenceResponse:
    """Predict weekly returns using the current US PatchTST model.

    Symbols default to the current model's training metadata. When
    ``symbols`` is provided in the request, inference runs only on that
    list (same model weights).

    Raises:
        HTTPException 400: if no current model version is available
        HTTPException 503: if model artifacts cannot be loaded
    """
    return _run_patchtst_inference(request, storage, log_prefix="[PatchTST]")


@router.post("/patchtst/india", response_model=PatchTSTInferenceResponse)
def infer_patchtst_india(
    request: PatchTSTInferenceRequest,
    storage: PatchTSTIndiaModelStorage = Depends(get_patchtst_india_storage),
) -> PatchTSTInferenceResponse:
    """Predict weekly returns using the current India PatchTST model.

    Same forward-pass math as the US route; the only difference is the
    storage backend (``data/models/patchtst_india/``). Symbols default to
    the India model's training metadata when not provided.

    Raises:
        HTTPException 400: if no current India model version is available
        HTTPException 503: if model artifacts cannot be loaded
    """
    return _run_patchtst_inference(request, storage, log_prefix="[PatchTST India]")


@router.post("/patchtst/score-batch", response_model=PatchTSTScoreBatchResponse)
def patchtst_score_batch(
    request: PatchTSTScoreBatchRequest,
    us_storage: PatchTSTModelStorage = Depends(get_patchtst_storage),
    india_storage: PatchTSTIndiaModelStorage = Depends(get_patchtst_india_storage),
) -> PatchTSTScoreBatchResponse:
    """Run PatchTST batch inference and apply rank-band score validation.

    Pipeline:

    1. Pick storage by ``request.market`` (``us`` -> US weights,
       ``india`` -> India weights). Reuses ``run_batch_inference`` --
       same forward-pass math for both markets.
    2. Apply ``validate_and_collect_finite_scores`` (math invariant for
       rank-band selection): exclude ``None`` predictions, raise on any
       non-finite value (NaN/+inf/-inf), enforce
       ``len(scores) >= min_predictions``.
    3. Return ``{symbol: predicted_weekly_return_pct}`` ready to feed
       ``/allocation/rank-band-top-n``.

    Raises:
        HTTPException 400: if no current model version is available for
            the requested market.
        HTTPException 422: if any prediction is non-finite, or if fewer
            than ``min_predictions`` finite scores are produced. These
            are math-invariant violations of the rank-band selector.
        HTTPException 503: if model artifacts cannot be loaded.
    """
    storage = india_storage if request.market == "india" else us_storage
    log_prefix = (
        "[PatchTST Score-Batch IN]"
        if request.market == "india"
        else "[PatchTST Score-Batch US]"
    )

    version = storage.read_current_version()
    if not version:
        raise HTTPException(
            400,
            f"No current PatchTST model version available for market={request.market}",
        )

    cutoff_date = get_patchtst_as_of_date(
        PatchTSTInferenceRequest(as_of_date=request.as_of_date)
    )
    week_boundaries = compute_week_from_cutoff(cutoff_date)

    symbols = list(request.symbols)
    logger.info(
        f"{log_prefix} Scoring {len(symbols)} symbols (model {version}, "
        f"as_of={cutoff_date})"
    )

    try:
        batch_result = run_batch_inference(symbols, cutoff_date, storage)
    except ValueError as e:
        raise HTTPException(503, str(e)) from e

    try:
        scores, excluded = validate_and_collect_finite_scores(
            batch_result.predictions,
            requested_count=len(symbols),
            min_predictions=request.min_predictions,
        )
    except RuntimeError as e:
        # Non-finite scores or below-floor count: math-invariant
        # violation of the rank-band selector. Return 422 (caller input
        # cannot be salvaged into a valid rank-band batch) rather than
        # 500.
        raise HTTPException(422, str(e)) from e

    logger.info(
        f"{log_prefix} {len(scores)} valid / {len(symbols)} requested, "
        f"excluded={len(excluded)}"
    )

    return PatchTSTScoreBatchResponse(
        scores=scores,
        model_version=batch_result.model_version,
        as_of_date=cutoff_date.isoformat(),
        target_week_start=week_boundaries.target_week_start.isoformat(),
        target_week_end=week_boundaries.target_week_end.isoformat(),
        requested_count=len(symbols),
        predicted_count=len(scores),
        excluded_symbols=excluded,
    )

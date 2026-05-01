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

from fastapi import APIRouter, HTTPException, Query

from brain_api.core.inference_utils import compute_week_from_cutoff
from brain_api.core.model_buckets import (
    BucketConfig,
    ModelType,
    UnknownBucketError,
    get_bucket,
    list_universes_for,
)
from brain_api.core.patchtst import validate_and_collect_finite_scores
from brain_api.core.patchtst.inference import run_batch_inference
from brain_api.storage.policy import load_current_artifacts_for_bucket

from .dependencies import (
    get_patchtst_as_of_date,
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
    bucket: BucketConfig,
    log_prefix: str,
) -> PatchTSTInferenceResponse:
    """Run a single-market PatchTST inference and shape the response.

    Shared between ``/patchtst`` (US) and ``/patchtst/india`` so the
    forward-pass + response-assembly math has one implementation -- only
    the bucket (trained weights/scalers) differs per market. Storage is
    resolved through the active policy (``local_first`` / ``hf_first``)
    so HF fallback works for both markets, closing audit Bug 1.
    """
    t_start = time.time()

    artifacts = load_current_artifacts_for_bucket(
        bucket=bucket,
        model_label=bucket.model_label,
    )
    version = artifacts.version
    storage = bucket.local_storage_class()

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
        batch_result = run_batch_inference(
            symbols, cutoff_date, storage=storage, artifacts=artifacts
        )
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


def _is_india_bucket(bucket: BucketConfig) -> bool:
    """Identify the India PatchTST bucket via its semantic name.

    We deliberately use ``bucket.bucket_name`` rather than
    ``isinstance``/identity on ``local_storage_class`` so test fixtures
    can swap the registry's ``local_storage_class`` (e.g. with a
    ``lambda`` factory returning a ``tmpdir``-backed storage instance)
    without breaking US/India routing. The bucket's name is preserved
    by ``dataclasses.replace`` and is the same identifier baked into
    the on-disk path and HF repo env var, so it is the right semantic
    key for "is this the India bucket?".
    """
    return bucket.bucket_name == "patchtst_nifty_shariah_500"


def _resolve_patchtst_us_bucket(universe: str | None) -> BucketConfig:
    """Resolve the bucket for a US PatchTST request.

    When ``universe`` is ``None`` we default to ``halal_new``. Any
    explicit value is looked up via the registry so future US buckets
    (e.g. ``patchtst_halal``) can be selected without endpoint
    signature changes.
    """
    resolved = universe if universe is not None else "halal_new"
    try:
        bucket = get_bucket(ModelType.PATCHTST, resolved)
    except UnknownBucketError as exc:
        allowed = sorted(list_universes_for(ModelType.PATCHTST))
        raise HTTPException(
            status_code=422,
            detail=(f"Unknown universe '{resolved}' for PatchTST. Allowed: {allowed}"),
        ) from exc
    if _is_india_bucket(bucket):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Universe '{resolved}' is an India bucket; use "
                "POST /inference/patchtst/india instead."
            ),
        )
    return bucket


def _resolve_patchtst_india_bucket(universe: str | None) -> BucketConfig:
    """Resolve the bucket for an India PatchTST request."""
    resolved = universe if universe is not None else "nifty_shariah_500"
    try:
        bucket = get_bucket(ModelType.PATCHTST, resolved)
    except UnknownBucketError as exc:
        allowed = sorted(list_universes_for(ModelType.PATCHTST))
        raise HTTPException(
            status_code=422,
            detail=(f"Unknown universe '{resolved}' for PatchTST. Allowed: {allowed}"),
        ) from exc
    if not _is_india_bucket(bucket):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Universe '{resolved}' is not an India bucket; use "
                "POST /inference/patchtst instead."
            ),
        )
    return bucket


@router.post("/patchtst", response_model=PatchTSTInferenceResponse)
def infer_patchtst(
    request: PatchTSTInferenceRequest,
    universe: str | None = Query(
        default=None,
        description=(
            "Optional bucket override. Defaults to the only registered US "
            "PatchTST universe (`halal_new`). Use this to point inference "
            "at a future bucket (e.g. `halal`) without breaking existing "
            "callers."
        ),
    ),
) -> PatchTSTInferenceResponse:
    """Predict weekly returns using the current US PatchTST model.

    Symbols default to the current model's training metadata. When
    ``symbols`` is provided in the request, inference runs only on that
    list (same model weights). Storage is resolved via the active
    policy (``local_first`` / ``hf_first``).

    Raises:
        HTTPException 422: if ``universe`` is unknown or refers to an India bucket
        HTTPException 503: if model artifacts cannot be loaded under the active policy
    """
    bucket = _resolve_patchtst_us_bucket(universe)
    return _run_patchtst_inference(request, bucket, log_prefix="[PatchTST]")


@router.post("/patchtst/india", response_model=PatchTSTInferenceResponse)
def infer_patchtst_india(
    request: PatchTSTInferenceRequest,
    universe: str | None = Query(
        default=None,
        description=(
            "Optional bucket override. Defaults to the only registered "
            "India PatchTST universe (`nifty_shariah_500`)."
        ),
    ),
) -> PatchTSTInferenceResponse:
    """Predict weekly returns using the current India PatchTST model.

    Same forward-pass math as the US route; the only difference is the
    bucket (``patchtst_nifty_shariah_500``). Symbols default to the
    India model's training metadata when not provided. Storage is
    resolved via the active policy (``local_first`` / ``hf_first``).

    Raises:
        HTTPException 422: if ``universe`` is unknown or refers to a US bucket
        HTTPException 503: if model artifacts cannot be loaded under the active policy
    """
    bucket = _resolve_patchtst_india_bucket(universe)
    return _run_patchtst_inference(request, bucket, log_prefix="[PatchTST India]")


@router.post("/patchtst/score-batch", response_model=PatchTSTScoreBatchResponse)
def patchtst_score_batch(
    request: PatchTSTScoreBatchRequest,
) -> PatchTSTScoreBatchResponse:
    """Run PatchTST batch inference and apply rank-band score validation.

    Pipeline:

    1. Pick the bucket by ``request.market`` (``us`` ->
       ``patchtst_halal_new``, ``india`` ->
       ``patchtst_nifty_shariah_500``). Reuses ``run_batch_inference``
       -- same forward-pass math for both markets.
    2. Apply ``validate_and_collect_finite_scores`` (math invariant for
       rank-band selection): exclude ``None`` predictions, raise on any
       non-finite value (NaN/+inf/-inf), enforce
       ``len(scores) >= min_predictions``.
    3. Return ``{symbol: predicted_weekly_return_pct}`` ready to feed
       ``/allocation/rank-band-top-n``.

    Raises:
        HTTPException 422: if any prediction is non-finite, or if fewer
            than ``min_predictions`` finite scores are produced. These
            are math-invariant violations of the rank-band selector.
        HTTPException 503: if model artifacts cannot be loaded under
            the active storage policy.
    """
    if request.market == "india":
        bucket = _resolve_patchtst_india_bucket(None)
        log_prefix = "[PatchTST Score-Batch IN]"
    else:
        bucket = _resolve_patchtst_us_bucket(None)
        log_prefix = "[PatchTST Score-Batch US]"

    artifacts = load_current_artifacts_for_bucket(
        bucket=bucket,
        model_label=bucket.model_label,
    )
    version = artifacts.version
    storage = bucket.local_storage_class()

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
        batch_result = run_batch_inference(
            symbols, cutoff_date, storage=storage, artifacts=artifacts
        )
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

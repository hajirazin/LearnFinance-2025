"""Generic inference pipeline for model predictions.

This module provides a base inference workflow that can be customized
for different model types (LSTM, PatchTST, etc.) while keeping the
common orchestration logic DRY.
"""

import logging
import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Generic, Protocol, TypeVar

from fastapi import HTTPException

from brain_api.core.config import get_hf_model_repo, get_storage_backend

logger = logging.getLogger(__name__)

# Type variables for generic typing
ConfigT = TypeVar("ConfigT")
ArtifactsT = TypeVar("ArtifactsT")
FeaturesT = TypeVar("FeaturesT")
PredictionT = TypeVar("PredictionT")


class LocalModelStorage(Protocol):
    """Protocol for local model storage backends."""

    def load_current_artifacts(self) -> Any: ...


@dataclass
class InferenceContext:
    """Context for an inference request."""

    model_name: str
    symbols: list[str]
    as_of_date: date
    target_week_start: date
    target_week_end: date
    data_start: date
    data_end: date


def log_timing(model_name: str, step: str, duration: float) -> None:
    """Log timing information for an inference step."""
    logger.info(f"[{model_name}] {step} in {duration:.2f}s")


def compute_data_window(
    target_week_start: date,
    context_length: int,
    buffer_multiplier: int = 2,
    extra_days: int = 30,
) -> tuple[date, date]:
    """Compute the data fetch window for inference.
    
    Args:
        target_week_start: First trading day of target prediction week
        context_length: Model's required history length (in trading days)
        buffer_multiplier: Multiplier for weekends/holidays
        extra_days: Additional safety buffer
        
    Returns:
        Tuple of (data_start, data_end) dates
    """
    buffer_days = context_length * buffer_multiplier + extra_days
    data_start = target_week_start - timedelta(days=buffer_days)
    data_end = target_week_start - timedelta(days=1)
    return data_start, data_end


def load_model_with_fallback(
    model_name: str,
    local_storage: LocalModelStorage,
    hf_storage_class: type,
) -> Any:
    """Load model artifacts with HuggingFace fallback.

    Tries local storage first, then falls back to HuggingFace Hub.

    Args:
        model_name: Model type identifier for logging
        local_storage: Local storage instance for caching
        hf_storage_class: HuggingFace storage class for fallback

    Returns:
        Model artifacts ready for inference

    Raises:
        HTTPException 503: if no model is available from any source
    """
    # Try local storage first
    try:
        return local_storage.load_current_artifacts()
    except (ValueError, FileNotFoundError) as local_error:
        logger.info(f"[{model_name}] Local model not found: {local_error}")

    # Try HuggingFace if configured
    storage_backend = get_storage_backend()
    hf_model_repo = get_hf_model_repo()

    if storage_backend == "hf" or hf_model_repo:
        if hf_model_repo:
            try:
                logger.info(
                    f"[{model_name}] Attempting to load model from HuggingFace: {hf_model_repo}"
                )
                hf_storage = hf_storage_class(
                    repo_id=hf_model_repo,
                    local_cache=local_storage,
                )
                return hf_storage.download_model(use_cache=True)
            except Exception as hf_error:
                logger.error(
                    f"[{model_name}] Failed to load model from HuggingFace: {hf_error}"
                )
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"No {model_name} model available. Local: model not trained. "
                        f"HuggingFace ({hf_model_repo}): {hf_error}"
                    ),
                ) from None

    # No model available from any source
    raise HTTPException(
        status_code=503,
        detail=(
            f"No trained {model_name} model available. "
            f"Train a model first with POST /train/{model_name.lower()}"
        ),
    ) from None


def sort_predictions_by_return(
    predictions: list[PredictionT],
    return_attr: str = "predicted_weekly_return_pct",
) -> list[PredictionT]:
    """Sort predictions by predicted return descending, nulls last.

    Args:
        predictions: List of prediction objects
        return_attr: Attribute name for the return value

    Returns:
        Sorted list with highest returns first, nulls at end
    """
    valid = [p for p in predictions if getattr(p, return_attr) is not None]
    invalid = [p for p in predictions if getattr(p, return_attr) is None]

    valid_sorted = sorted(
        valid,
        key=lambda p: getattr(p, return_attr),
        reverse=True,
    )

    return valid_sorted + invalid


@dataclass
class InferenceOutcome(Generic[PredictionT]):
    """Result of inference pipeline execution."""

    predictions: list[PredictionT]
    model_version: str
    as_of_date: str
    target_week_start: str
    target_week_end: str


def log_inference_summary(
    model_name: str,
    predictions: list[Any],
    total_symbols: int,
    total_time: float,
    return_attr: str = "predicted_weekly_return_pct",
) -> None:
    """Log summary of inference results."""
    valid_predictions = [
        p for p in predictions if getattr(p, return_attr) is not None
    ]
    logger.info(
        f"[{model_name}] Request complete: {len(valid_predictions)}/{total_symbols} "
        f"predictions in {total_time:.2f}s"
    )

    if valid_predictions:
        top = valid_predictions[0]
        bottom = valid_predictions[-1]
        top_return = getattr(top, return_attr)
        bottom_return = getattr(bottom, return_attr)
        logger.info(
            f"[{model_name}] Top: {top.symbol} ({top_return:+.2f}%), "
            f"Bottom: {bottom.symbol} ({bottom_return:+.2f}%)"
        )


"""Helper functions for inference endpoints."""

import logging
from datetime import date

from fastapi import HTTPException

from brain_api.core.config import (
    get_hf_lstm_model_repo,
    get_hf_patchtst_model_repo,
    get_storage_backend,
)
from brain_api.core.lstm import SymbolPrediction as LSTMSymbolPrediction
from brain_api.core.patchtst import SymbolPrediction as PatchTSTSymbolPrediction
from brain_api.core.realtime_signals import (
    LSTMForecaster,
    PatchTSTForecaster,
    RealTimeSignalBuilder,
)
from brain_api.storage.local import (
    LocalModelStorage,
    LSTMArtifacts,
    PatchTSTArtifacts,
    PatchTSTModelStorage,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Real-time signal and forecast helpers for RL inference
# =============================================================================

# Module-level instances (lazy initialization)
_signal_builder: RealTimeSignalBuilder | None = None
_lstm_forecaster: LSTMForecaster | None = None
_patchtst_forecaster: PatchTSTForecaster | None = None


def _get_signal_builder() -> RealTimeSignalBuilder:
    """Get or create the signal builder instance."""
    global _signal_builder
    if _signal_builder is None:
        _signal_builder = RealTimeSignalBuilder()
    return _signal_builder


def _get_lstm_forecaster() -> LSTMForecaster:
    """Get or create the LSTM forecaster instance."""
    global _lstm_forecaster
    if _lstm_forecaster is None:
        _lstm_forecaster = LSTMForecaster()
    return _lstm_forecaster


def _get_patchtst_forecaster() -> PatchTSTForecaster:
    """Get or create the PatchTST forecaster instance."""
    global _patchtst_forecaster
    if _patchtst_forecaster is None:
        _patchtst_forecaster = PatchTSTForecaster()
    return _patchtst_forecaster


def build_current_signals(
    symbols: list[str],
    as_of_date: date,
) -> dict[str, dict[str, float]]:
    """Build current signals for RL inference using real-time data.

    Delegates to RealTimeSignalBuilder for fetching:
    - News sentiment: yfinance news + FinBERT scoring
    - Fundamentals: yfinance ticker.info (gross_margin, operating_margin, etc.)

    Args:
        symbols: List of stock ticker symbols
        as_of_date: Reference date for fetching data

    Returns:
        Dict mapping symbol -> dict of signal values matching training format
    """
    builder = _get_signal_builder()
    return builder.build(symbols, as_of_date)


def build_current_forecasts(
    symbols: list[str],
    forecaster_type: str,
    as_of_date: date,
) -> dict[str, float]:
    """Build current forecast features for RL inference.

    Delegates to LSTMForecaster or PatchTSTForecaster.

    Args:
        symbols: List of stock ticker symbols
        forecaster_type: "lstm" or "patchtst"
        as_of_date: Reference date for inference

    Returns:
        Dict mapping symbol -> predicted weekly return (decimal)
    """
    if forecaster_type == "lstm":
        forecaster = _get_lstm_forecaster()
    elif forecaster_type == "patchtst":
        forecaster = _get_patchtst_forecaster()
    else:
        logger.warning(f"[Forecasts] Unknown forecaster type: {forecaster_type}")
        return dict.fromkeys(symbols, 0.0)

    return forecaster.build_forecasts(symbols, as_of_date)


def _load_model_artifacts_generic(
    model_type: str,
    local_storage: LocalModelStorage | PatchTSTModelStorage,
    hf_storage_class: type,
    hf_model_repo: str | None,
) -> LSTMArtifacts | PatchTSTArtifacts:
    """Load model artifacts with HuggingFace fallback.

    Generic helper that handles local → HuggingFace fallback for any model type.

    Args:
        model_type: Model type identifier (e.g., "LSTM", "PatchTST", "PPO", "SAC")
        local_storage: Local storage instance for caching
        hf_storage_class: HuggingFace storage class to use for fallback
        hf_model_repo: HuggingFace repository ID for this model type

    Returns:
        Model artifacts ready for inference

    Raises:
        HTTPException 503: if no model is available from any source
    """
    # Try local storage first
    try:
        return local_storage.load_current_artifacts()
    except (ValueError, FileNotFoundError) as local_error:
        logger.info(f"[{model_type}] Local model not found: {local_error}")

    # Try HuggingFace if configured
    storage_backend = get_storage_backend()

    if storage_backend == "hf" or hf_model_repo:
        if hf_model_repo:
            try:
                logger.info(
                    f"[{model_type}] Attempting to load model from HuggingFace: {hf_model_repo}"
                )
                hf_storage = hf_storage_class(
                    repo_id=hf_model_repo,
                    local_cache=local_storage,
                )
                return hf_storage.download_model(use_cache=True)
            except Exception as hf_error:
                logger.error(
                    f"[{model_type}] Failed to load model from HuggingFace: {hf_error}"
                )
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"No {model_type} model available. Local: model not trained. "
                        f"HuggingFace ({hf_model_repo}): {hf_error}"
                    ),
                ) from None

    # No model available from any source
    raise HTTPException(
        status_code=503,
        detail=f"No trained {model_type} model available. Train a model first with POST /train/{model_type.lower()}",
    ) from None


def _load_model_artifacts(storage: LocalModelStorage) -> LSTMArtifacts:
    """Load LSTM model artifacts with HuggingFace fallback."""
    from brain_api.storage.huggingface import HuggingFaceModelStorage

    return _load_model_artifacts_generic(
        "LSTM", storage, HuggingFaceModelStorage, get_hf_lstm_model_repo()
    )


def _load_patchtst_model_artifacts(storage: PatchTSTModelStorage) -> PatchTSTArtifacts:
    """Load PatchTST model artifacts with HuggingFace fallback."""
    from brain_api.storage.huggingface import PatchTSTHuggingFaceModelStorage

    return _load_model_artifacts_generic(
        "PatchTST",
        storage,
        PatchTSTHuggingFaceModelStorage,
        get_hf_patchtst_model_repo(),
    )


def _sort_predictions(
    predictions: list[LSTMSymbolPrediction],
) -> list[LSTMSymbolPrediction]:
    """Sort predictions by predicted_weekly_return_pct descending.

    Predictions with valid returns are sorted highest to lowest.
    Predictions with null returns (insufficient history) are placed at the end.
    """
    # Separate valid and invalid predictions
    valid = [p for p in predictions if p.predicted_weekly_return_pct is not None]
    invalid = [p for p in predictions if p.predicted_weekly_return_pct is None]

    # Sort valid predictions by return (highest first)
    valid_sorted = sorted(
        valid,
        key=lambda p: p.predicted_weekly_return_pct,  # type: ignore[arg-type]
        reverse=True,
    )

    return valid_sorted + invalid


def _sort_patchtst_predictions(
    predictions: list[PatchTSTSymbolPrediction],
) -> list[PatchTSTSymbolPrediction]:
    """Sort PatchTST predictions by predicted_weekly_return_pct descending."""
    valid = [p for p in predictions if p.predicted_weekly_return_pct is not None]
    invalid = [p for p in predictions if p.predicted_weekly_return_pct is None]

    valid_sorted = sorted(
        valid,
        key=lambda p: p.predicted_weekly_return_pct,  # type: ignore[arg-type]
        reverse=True,
    )

    return valid_sorted + invalid

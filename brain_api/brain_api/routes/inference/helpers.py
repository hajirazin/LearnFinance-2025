"""Helper functions for inference endpoints."""

import logging
from datetime import date

from brain_api.core.lstm import SymbolPrediction as LSTMSymbolPrediction
from brain_api.core.patchtst import SymbolPrediction as PatchTSTSymbolPrediction
from brain_api.core.realtime_signals import (
    LSTMForecaster,
    PatchTSTForecaster,
    RealTimeSignalBuilder,
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

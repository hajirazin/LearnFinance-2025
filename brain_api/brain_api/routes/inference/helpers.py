"""Helper functions for inference endpoints."""

from brain_api.core.lstm import SymbolPrediction as LSTMSymbolPrediction
from brain_api.core.patchtst import SymbolPrediction as PatchTSTSymbolPrediction


def _sort_predictions(
    predictions: list[LSTMSymbolPrediction],
) -> list[LSTMSymbolPrediction]:
    """Sort predictions by predicted_weekly_return_pct descending.

    Predictions with valid returns are sorted highest to lowest.
    Predictions with null returns (insufficient history) are placed at the end.
    """
    valid = [p for p in predictions if p.predicted_weekly_return_pct is not None]
    invalid = [p for p in predictions if p.predicted_weekly_return_pct is None]
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

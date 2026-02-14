"""LSTM inference helpers."""

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from brain_api.core.features import compute_ohlcv_log_returns
from brain_api.core.inference_utils import WeekBoundaries
from brain_api.core.lstm.config import LSTMConfig
from brain_api.core.lstm.model import LSTMModel
from brain_api.core.model_types import classify_direction


@dataclass
class InferenceFeatures:
    """Features prepared for inference for a single symbol."""

    symbol: str
    features: (
        np.ndarray | None
    )  # Shape: (seq_len, n_features) or None if insufficient data
    has_enough_history: bool
    history_days_used: int
    data_end_date: (
        date | None
    )  # Last date of data used (should be before target_week_start)


@dataclass
class SymbolPrediction:
    """Prediction result for a single symbol."""

    symbol: str
    predicted_weekly_return_pct: float | None  # Percentage (e.g., 2.5 for +2.5%)
    predicted_volatility: float | None  # Std dev of 5 daily return predictions
    direction: str  # "UP", "DOWN", or "FLAT"
    has_enough_history: bool
    history_days_used: int
    data_end_date: str | None  # ISO format
    target_week_start: str  # ISO format
    target_week_end: str  # ISO format
    daily_returns: list[float] | None = None  # 5 predicted daily close log returns


def build_inference_features(
    symbol: str,
    prices_df: pd.DataFrame,
    config: LSTMConfig,
    cutoff_date: date,
) -> InferenceFeatures:
    """Build feature sequence for inference from OHLCV data.

    Constructs the same log-return features as training, ending just before
    cutoff_date (which should be target_week_start).

    Args:
        symbol: Ticker symbol
        prices_df: DataFrame with OHLCV columns (open, high, low, close, volume)
                   and DatetimeIndex
        config: LSTM config with sequence_length and use_returns settings
        cutoff_date: Features end before this date (typically target_week_start)

    Returns:
        InferenceFeatures with prepared feature sequence or None if insufficient data
    """
    if prices_df.empty:
        return InferenceFeatures(
            symbol=symbol,
            features=None,
            has_enough_history=False,
            history_days_used=0,
            data_end_date=None,
        )

    # Ensure DatetimeIndex
    if not isinstance(prices_df.index, pd.DatetimeIndex):
        return InferenceFeatures(
            symbol=symbol,
            features=None,
            has_enough_history=False,
            history_days_used=0,
            data_end_date=None,
        )

    # Filter to data before cutoff_date
    cutoff_ts = pd.Timestamp(cutoff_date)
    if prices_df.index.tz is not None:
        cutoff_ts = cutoff_ts.tz_localize(prices_df.index.tz)
    df = prices_df[prices_df.index < cutoff_ts].copy()

    if len(df) < config.sequence_length + 1:  # +1 for the shift in log returns
        return InferenceFeatures(
            symbol=symbol,
            features=None,
            has_enough_history=False,
            history_days_used=len(df),
            data_end_date=df.index[-1].date() if len(df) > 0 else None,
        )

    # Compute features using shared utility
    features_df = compute_ohlcv_log_returns(df, use_returns=config.use_returns)

    # Check if we have enough data after computing returns
    if len(features_df) < config.sequence_length:
        return InferenceFeatures(
            symbol=symbol,
            features=None,
            has_enough_history=False,
            history_days_used=len(features_df),
            data_end_date=features_df.index[-1].date()
            if len(features_df) > 0
            else None,
        )

    # Take the last sequence_length rows
    sequence = features_df.iloc[-config.sequence_length :].values
    data_end_date = features_df.index[-1].date()

    return InferenceFeatures(
        symbol=symbol,
        features=sequence,
        has_enough_history=True,
        history_days_used=len(features_df),
        data_end_date=data_end_date,
    )


def run_inference(
    model: LSTMModel,
    feature_scaler: StandardScaler,
    features_list: list[InferenceFeatures],
    week_boundaries: WeekBoundaries,
) -> list[SymbolPrediction]:
    """Run LSTM inference on prepared feature sequences.

    Single forward pass predicts 5 daily close log returns directly.
    Weekly return = exp(sum(5 log returns)) - 1.
    Volatility = std of the 5 daily predictions.

    NO inverse transform is applied -- targets are never scaled during training
    (dataset.py keeps returns in original scale), so the model's linear layer
    outputs raw log returns directly.

    Args:
        model: Loaded LSTMModel in eval mode
        feature_scaler: Fitted StandardScaler from training
        features_list: List of InferenceFeatures (one per symbol)
        week_boundaries: Target week info for the response

    Returns:
        List of SymbolPrediction results
    """
    predictions = []

    # Separate symbols with/without sufficient data
    valid_features = [(f.symbol, f) for f in features_list if f.features is not None]
    invalid_features = [f for f in features_list if f.features is None]

    # Handle symbols without enough data
    for feat in invalid_features:
        predictions.append(
            SymbolPrediction(
                symbol=feat.symbol,
                predicted_weekly_return_pct=None,
                predicted_volatility=None,
                direction="FLAT",
                has_enough_history=False,
                history_days_used=feat.history_days_used,
                data_end_date=feat.data_end_date.isoformat()
                if feat.data_end_date
                else None,
                target_week_start=week_boundaries.target_week_start.isoformat(),
                target_week_end=week_boundaries.target_week_end.isoformat(),
            )
        )

    if not valid_features:
        return predictions

    # Batch inference for valid symbols
    # Shape: (n_samples, seq_len, n_features)
    X_batch = np.array([f.features for _, f in valid_features])

    # Scale features using the training scaler
    original_shape = X_batch.shape
    X_flat = X_batch.reshape(-1, X_batch.shape[-1])
    X_scaled = feature_scaler.transform(X_flat)
    X_batch = X_scaled.reshape(original_shape)

    # Single forward pass: model outputs (batch, 5) = 5 close log returns
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_batch)
        outputs = model(X_tensor)
        daily_preds = outputs.cpu().numpy()  # (batch, 5)

    # NO inverse transform -- Bug #1 fix.
    # Targets are never scaled (dataset.py keeps returns in original scale).
    # Model linear layer outputs raw log returns directly.

    # Build prediction results
    for i, (symbol, feat) in enumerate(valid_features):
        symbol_daily = daily_preds[i]  # (5,) = 5 close log returns

        # Weekly return by compounding log returns: exp(sum) - 1
        weekly_return = float(np.exp(np.sum(symbol_daily)) - 1)

        # Volatility as std dev of daily log return predictions
        volatility = float(np.std(symbol_daily))

        weekly_return_pct = weekly_return * 100  # Convert to percentage
        direction = classify_direction(weekly_return)

        predictions.append(
            SymbolPrediction(
                symbol=symbol,
                predicted_weekly_return_pct=round(weekly_return_pct, 4),
                predicted_volatility=round(volatility, 6),
                direction=direction,
                has_enough_history=True,
                history_days_used=feat.history_days_used,
                data_end_date=feat.data_end_date.isoformat()
                if feat.data_end_date
                else None,
                target_week_start=week_boundaries.target_week_start.isoformat(),
                target_week_end=week_boundaries.target_week_end.isoformat(),
                daily_returns=[round(float(r), 6) for r in symbol_daily],
            )
        )

    return predictions

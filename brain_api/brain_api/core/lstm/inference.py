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

    Predicts 5 daily returns iteratively (Monday-Friday) and aggregates to weekly return.
    This matches the walk-forward training approach and enables volatility computation.

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

    # OHLCV feature indices: open_ret=0, high_ret=1, low_ret=2, close_ret=3, volume_ret=4
    close_ret_idx = 3
    open_ret_idx = 0
    high_ret_idx = 1
    low_ret_idx = 2
    volume_ret_idx = 4

    # Predict 5 daily returns iteratively (Monday through Friday)
    n_samples = len(valid_features)
    daily_returns = np.zeros((5, n_samples), dtype=np.float32)

    model.eval()
    X_current = X_batch.copy()  # Working copy that we'll update each iteration

    with torch.no_grad():
        for day in range(5):
            # Convert to tensor and run model
            X_tensor = torch.FloatTensor(X_current)
            outputs = model(X_tensor)
            scaled_preds = outputs.cpu().numpy().flatten()

            # Inverse transform predictions from scaled space back to return space
            # Model outputs are in StandardScaler space: unscaled = scaled * std + mean
            # For LSTM trained on daily returns, output is close_ret prediction
            daily_returns[day] = (
                scaled_preds * feature_scaler.scale_[close_ret_idx]
                + feature_scaler.mean_[close_ret_idx]
            )

            # Clean up tensor
            del X_tensor, outputs

            # Update input sequences for next iteration (except for the last day)
            if day < 4:
                # Construct new day features
                # Set all OHLCV returns to predicted close_ret (simplified assumption)
                pred_returns = daily_returns[day].reshape(-1, 1)

                # Scale the predicted returns back to StandardScaler space
                scaled_pred = (
                    pred_returns - feature_scaler.mean_[close_ret_idx]
                ) / feature_scaler.scale_[close_ret_idx]

                # Create new day features array
                new_day_features = np.zeros((n_samples, 5), dtype=np.float32)
                new_day_features[:, open_ret_idx] = scaled_pred.ravel()
                new_day_features[:, high_ret_idx] = scaled_pred.ravel()
                new_day_features[:, low_ret_idx] = scaled_pred.ravel()
                new_day_features[:, close_ret_idx] = scaled_pred.ravel()
                # Volume return assumed 0 (scaled)
                new_day_features[:, volume_ret_idx] = (
                    -feature_scaler.mean_[volume_ret_idx]
                    / feature_scaler.scale_[volume_ret_idx]
                )

                # Update sequences: shift left and add new day
                X_current[:, :-1, :] = X_current[:, 1:, :]
                X_current[:, -1, :] = new_day_features

    # Build prediction results
    for i, (symbol, feat) in enumerate(valid_features):
        # Get 5 daily returns for this symbol
        symbol_daily_returns = daily_returns[:, i]  # Shape: (5,)

        # Compute weekly return by compounding daily returns
        weekly_return = float(np.prod(1 + symbol_daily_returns) - 1)

        # Compute volatility as std dev of daily returns
        volatility = float(np.std(symbol_daily_returns))

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
            )
        )

    return predictions

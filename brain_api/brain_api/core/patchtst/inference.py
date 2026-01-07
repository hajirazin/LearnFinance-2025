"""PatchTST inference helpers."""

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from transformers import PatchTSTForPrediction

from brain_api.core.features import compute_ohlcv_log_returns
from brain_api.core.inference_utils import WeekBoundaries
from brain_api.core.model_types import classify_direction
from brain_api.core.patchtst.config import PatchTSTConfig


@dataclass
class InferenceFeatures:
    """Multi-channel features prepared for inference for a single symbol."""

    symbol: str
    features: np.ndarray | None  # Shape: (context_length, n_channels) or None
    has_enough_history: bool
    history_days_used: int
    data_end_date: date | None
    has_news_data: bool
    has_fundamentals_data: bool
    starting_price: float | None  # Starting price (last close before target week) for weekly return calculation


@dataclass
class SymbolPrediction:
    """Prediction result for a single symbol."""

    symbol: str
    predicted_weekly_return_pct: float | None
    direction: str  # "UP", "DOWN", or "FLAT"
    has_enough_history: bool
    history_days_used: int
    data_end_date: str | None
    target_week_start: str
    target_week_end: str
    has_news_data: bool
    has_fundamentals_data: bool


def build_inference_features(
    symbol: str,
    prices_df: pd.DataFrame,
    news_df: pd.DataFrame | None,
    fundamentals_df: pd.DataFrame | None,
    config: PatchTSTConfig,
    cutoff_date: date,
) -> InferenceFeatures:
    """Build multi-channel feature sequence for inference.

    Constructs the same features as training:
    - OHLCV log returns (5 channels)
    - News sentiment (1 channel) - forward-filled
    - Fundamentals (5 channels) - forward-filled

    Args:
        symbol: Ticker symbol
        prices_df: DataFrame with OHLCV columns and DatetimeIndex
        news_df: DataFrame with 'sentiment_score' column and DatetimeIndex (or None)
        fundamentals_df: DataFrame with fundamental ratio columns and DatetimeIndex (or None)
        config: PatchTST config with context_length and feature settings
        cutoff_date: Features end before this date (typically target_week_start)

    Returns:
        InferenceFeatures with prepared multi-channel feature sequence
    """
    has_news_data = news_df is not None and len(news_df) > 0
    has_fundamentals_data = fundamentals_df is not None and len(fundamentals_df) > 0

    if prices_df.empty:
        return InferenceFeatures(
            symbol=symbol,
            features=None,
            has_enough_history=False,
            history_days_used=0,
            data_end_date=None,
            has_news_data=has_news_data,
            has_fundamentals_data=has_fundamentals_data,
            starting_price=None,
        )

    if not isinstance(prices_df.index, pd.DatetimeIndex):
        return InferenceFeatures(
            symbol=symbol,
            features=None,
            has_enough_history=False,
            history_days_used=0,
            data_end_date=None,
            has_news_data=has_news_data,
            has_fundamentals_data=has_fundamentals_data,
            starting_price=None,
        )

    # Filter to data before cutoff
    cutoff_ts = pd.Timestamp(cutoff_date)
    df = prices_df[prices_df.index < cutoff_ts].copy()

    if len(df) < config.context_length + 1:
            return InferenceFeatures(
                symbol=symbol,
                features=None,
                has_enough_history=False,
                history_days_used=len(df),
                data_end_date=df.index[-1].date() if len(df) > 0 else None,
                has_news_data=has_news_data,
                has_fundamentals_data=has_fundamentals_data,
                starting_price=None,
            )

    # Compute price features using shared utility
    features_df = compute_ohlcv_log_returns(df, use_returns=config.use_returns)

    if len(features_df) < config.context_length:
        return InferenceFeatures(
            symbol=symbol,
            features=None,
            has_enough_history=False,
            history_days_used=len(features_df),
            data_end_date=features_df.index[-1].date() if len(features_df) > 0 else None,
            has_news_data=has_news_data,
            has_fundamentals_data=has_fundamentals_data,
            starting_price=None,
        )

    # Add news sentiment (forward-fill missing days)
    if news_df is not None and len(news_df) > 0:
        sentiment_aligned = news_df.reindex(features_df.index, method="ffill")
        features_df["news_sentiment"] = sentiment_aligned["sentiment_score"].fillna(0.0)
    else:
        features_df["news_sentiment"] = 0.0  # Neutral if no news data

    # Add fundamentals (forward-fill quarterly data)
    fundamental_cols = ["gross_margin", "operating_margin", "net_margin",
                       "current_ratio", "debt_to_equity"]
    if fundamentals_df is not None and len(fundamentals_df) > 0:
        fund_aligned = fundamentals_df.reindex(features_df.index, method="ffill")
        
        # Calculate days since last fundamental update for each date
        fundamental_age_days = []
        for date in features_df.index:
            # Find the most recent fundamental date <= current date
            available_dates = fundamentals_df.index[fundamentals_df.index <= date]
            if len(available_dates) > 0:
                last_update = available_dates[-1]
                days_old = (date - last_update).days
                fundamental_age_days.append(days_old)
            else:
                fundamental_age_days.append(999)  # Very old if no data
        
        # Normalize age: 0.0 = fresh (0 days), 1.0 = 90 days old (quarterly)
        features_df["fundamental_age"] = pd.Series(fundamental_age_days, index=features_df.index) / 90.0
        
        for col in fundamental_cols:
            if col in fund_aligned.columns:
                features_df[col] = fund_aligned[col].fillna(0.0)
            else:
                features_df[col] = 0.0
    else:
        # No fundamentals - use zeros and max age
        features_df["fundamental_age"] = 1.0  # Max age (90+ days)
        for col in fundamental_cols:
            features_df[col] = 0.0

    # Ensure column order matches config.feature_names
    features_df = features_df[config.feature_names]

    # Take last context_length rows
    sequence = features_df.iloc[-config.context_length:].values
    data_end_date = features_df.index[-1].date()

    # Get starting price: last close price before cutoff_date (for weekly return calculation)
    # This will be used as the base price for computing weekly return from daily predictions
    starting_price = None
    if len(df) > 0:
        try:
            # Get the last close price before cutoff_date
            last_close = df.iloc[-1]["close"]
            if pd.notna(last_close) and last_close > 0:
                starting_price = float(last_close)
        except (KeyError, IndexError):
            pass

    return InferenceFeatures(
        symbol=symbol,
        features=sequence,
        has_enough_history=True,
        history_days_used=len(features_df),
        data_end_date=data_end_date,
        has_news_data=has_news_data,
        has_fundamentals_data=has_fundamentals_data,
        starting_price=starting_price,
    )


def run_inference(
    model: PatchTSTForPrediction,
    feature_scaler: StandardScaler,
    features_list: list[InferenceFeatures],
    week_boundaries: WeekBoundaries,
    config: PatchTSTConfig,
) -> list[SymbolPrediction]:
    """Run PatchTST inference on prepared multi-channel feature sequences.

    Predicts 5 daily returns iteratively (Monday-Friday) and aggregates to weekly return.

    Args:
        model: Loaded PatchTSTForPrediction model in eval mode
        feature_scaler: Fitted StandardScaler from training
        features_list: List of InferenceFeatures (one per symbol)
        week_boundaries: Target week info for the response
        config: PatchTST configuration (for feature names)

    Returns:
        List of SymbolPrediction results
    """
    predictions = []

    valid_features = [(f.symbol, f) for f in features_list if f.features is not None]
    invalid_features = [f for f in features_list if f.features is None]

    # Handle symbols without enough data
    for feat in invalid_features:
        predictions.append(
            SymbolPrediction(
                symbol=feat.symbol,
                predicted_weekly_return_pct=None,
                direction="FLAT",
                has_enough_history=False,
                history_days_used=feat.history_days_used,
                data_end_date=feat.data_end_date.isoformat() if feat.data_end_date else None,
                target_week_start=week_boundaries.target_week_start.isoformat(),
                target_week_end=week_boundaries.target_week_end.isoformat(),
                has_news_data=feat.has_news_data,
                has_fundamentals_data=feat.has_fundamentals_data,
            )
        )

    if not valid_features:
        return predictions

    # Find close_ret channel index
    try:
        close_ret_idx = config.feature_names.index("close_ret")
    except ValueError:
        raise ValueError(f"close_ret not found in feature_names: {config.feature_names}")

    # Prepare initial input sequences for all symbols
    # Shape: (n_samples, context_length, num_channels)
    X_batch = np.array([f.features for _, f in valid_features])
    
    # Scale features using the training scaler
    original_shape = X_batch.shape
    X_flat = X_batch.reshape(-1, X_batch.shape[-1])
    X_scaled = feature_scaler.transform(X_flat)
    X_batch = X_scaled.reshape(original_shape)

    model.eval()
    device = next(model.parameters()).device
    
    # Find indices for OHLCV channels (for constructing new day's features)
    try:
        open_ret_idx = config.feature_names.index("open_ret")
        high_ret_idx = config.feature_names.index("high_ret")
        low_ret_idx = config.feature_names.index("low_ret")
        volume_ret_idx = config.feature_names.index("volume_ret")
    except ValueError as e:
        raise ValueError(f"Required channel not found in feature_names: {e}")

    # Predict 5 daily returns iteratively (Monday through Friday)
    # After each prediction, update the input sequence by:
    # 1. Removing the oldest day (first row)
    # 2. Adding the new day's features (constructed from prediction)
    daily_returns = []
    
    # Current input sequences (will be updated each iteration)
    X_current = X_batch.copy()  # Shape: (n_samples, context_length, num_channels)
    
    with torch.no_grad():
        for day in range(5):
            # Convert to tensor and move to device
            X_tensor = torch.FloatTensor(X_current).to(device)
            
            # Predict next-day return
            outputs = model(past_values=X_tensor).prediction_outputs
            # Extract close_ret channel for next-day return prediction
            day_returns = outputs[:, 0, close_ret_idx].cpu().numpy()  # Shape: (n_samples,)
            daily_returns.append(day_returns)
            
            # Update input sequences for next iteration (except for the last day)
            if day < 4:  # Don't need to update after Friday prediction
                # Construct features for the predicted day
                # We need to work in unscaled space, then scale everything together
                # Get the last day's features in unscaled space for forward-filling
                X_current_unscaled = feature_scaler.inverse_transform(
                    X_current.reshape(-1, config.num_input_channels)
                ).reshape(X_current.shape)
                
                new_day_features = np.zeros((len(valid_features), config.num_input_channels))
                
                for i in range(len(valid_features)):
                    # Use predicted close_ret for OHLCV channels (in unscaled space)
                    predicted_close_ret = day_returns[i]  # Already in unscaled space
                    new_day_features[i, open_ret_idx] = predicted_close_ret
                    new_day_features[i, high_ret_idx] = predicted_close_ret
                    new_day_features[i, low_ret_idx] = predicted_close_ret
                    new_day_features[i, close_ret_idx] = predicted_close_ret
                    new_day_features[i, volume_ret_idx] = 0.0  # Assume no volume change
                    
                    # Forward-fill other channels (news, fundamentals) from the last day
                    # These don't change day-to-day, so use the last row's unscaled values
                    last_day_features_unscaled = X_current_unscaled[i, -1, :]
                    for ch_idx in range(config.num_input_channels):
                        if ch_idx not in [open_ret_idx, high_ret_idx, low_ret_idx, 
                                         close_ret_idx, volume_ret_idx]:
                            new_day_features[i, ch_idx] = last_day_features_unscaled[ch_idx]
                
                # Scale the new day's features
                new_day_features_scaled = feature_scaler.transform(new_day_features)
                
                # Update sequences: remove oldest day, add new day
                # X_current shape: (n_samples, context_length, num_channels)
                # Remove first row, add new row at the end
                X_current = np.concatenate([
                    X_current[:, 1:, :],  # Remove oldest day (first row)
                    new_day_features_scaled[:, np.newaxis, :]  # Add new day (add dimension for sequence)
                ], axis=1)
    
    # Convert to numpy array: shape (5, n_samples)
    daily_returns = np.array(daily_returns)

    # Build prediction results
    for i, (symbol, feat) in enumerate(valid_features):
        # Get 5 daily returns for this symbol
        symbol_daily_returns = daily_returns[:, i]  # Shape: (5,)
        
        # Compute weekly return from 5 daily returns
        # Method 1: Compound returns: (1 + r1) * (1 + r2) * ... * (1 + r5) - 1
        weekly_return = float(np.prod(1 + symbol_daily_returns) - 1)
        
        # Alternative: If we have starting_price, compute final price and return
        # This is more accurate but requires starting_price
        if feat.starting_price is not None and feat.starting_price > 0:
            final_price = feat.starting_price * np.prod(1 + symbol_daily_returns)
            weekly_return = (final_price - feat.starting_price) / feat.starting_price
        
        weekly_return_pct = weekly_return * 100
        direction = classify_direction(weekly_return)

        predictions.append(
            SymbolPrediction(
                symbol=symbol,
                predicted_weekly_return_pct=round(weekly_return_pct, 4),
                direction=direction,
                has_enough_history=True,
                history_days_used=feat.history_days_used,
                data_end_date=feat.data_end_date.isoformat() if feat.data_end_date else None,
                target_week_start=week_boundaries.target_week_start.isoformat(),
                target_week_end=week_boundaries.target_week_end.isoformat(),
                has_news_data=feat.has_news_data,
                has_fundamentals_data=feat.has_fundamentals_data,
            )
        )

    return predictions

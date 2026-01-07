"""Dataset building for PatchTST training."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from brain_api.core.inference_utils import extract_trading_weeks
from brain_api.core.patchtst.config import PatchTSTConfig


@dataclass
class DatasetResult:
    """Result of dataset building for weekly return prediction."""

    X: np.ndarray  # Input sequences: (n_samples, context_length, n_channels)
    y: np.ndarray  # Targets: (n_samples, prediction_length) - weekly returns
    feature_scaler: StandardScaler  # Scaler for input features


def _compute_weekly_return(
    price_df: pd.DataFrame,
    week_start: pd.Timestamp,
    week_end: pd.Timestamp,
) -> float | None:
    """Compute weekly return from OHLCV data.

    Args:
        price_df: DataFrame with OHLCV columns and DatetimeIndex
        week_start: First trading day of the week
        week_end: Last trading day of the week

    Returns:
        Weekly return = (week_end_close - week_start_open) / week_start_open
        or None if data is missing
    """
    try:
        start_price = price_df.loc[week_start, "open"]
        end_price = price_df.loc[week_end, "close"]

        if start_price == 0 or pd.isna(start_price) or pd.isna(end_price):
            return None

        return (end_price - start_price) / start_price
    except KeyError:
        return None


def build_dataset(
    aligned_features: dict[str, pd.DataFrame],
    prices: dict[str, pd.DataFrame],
    config: PatchTSTConfig,
) -> DatasetResult:
    """Build training dataset for next-day return prediction.

    Creates samples for daily return prediction:
    - Input: context_length days of multi-channel features ending at day t
    - Target: next-day return (close_ret channel for day t+1)

    Args:
        aligned_features: Dict of symbol -> aligned multi-channel DataFrame
                         (output from align_multivariate_data)
        prices: Dict of symbol -> raw OHLCV DataFrame (not used, kept for compatibility)
        config: PatchTST configuration

    Returns:
        DatasetResult with X, y (next-day returns), and feature_scaler
    """
    all_sequences = []
    all_targets = []

    print(f"[PatchTST] Building dataset from {len(aligned_features)} symbols...")
    symbols_used = 0
    total_samples = 0

    # Find close_ret channel index
    try:
        close_ret_idx = config.feature_names.index("close_ret")
    except ValueError:
        raise ValueError(f"close_ret not found in feature_names: {config.feature_names}")

    for symbol, features_df in aligned_features.items():
        if len(features_df) < config.context_length + 1:
            continue

        symbol_samples = 0

        # Create samples: for each day t, predict day t+1's return
        # We need at least context_length days of history, and at least 1 day ahead
        for t in range(config.context_length, len(features_df) - 1):
            # Extract input sequence: days [t-context_length, t)
            seq_start_idx = t - config.context_length
            seq_end_idx = t

            sequence = features_df.iloc[seq_start_idx:seq_end_idx].values

            if len(sequence) != config.context_length:
                continue

            # Target: next-day close return (day t+1)
            next_day_close_ret = features_df.iloc[t + 1, close_ret_idx]

            # Skip if target is NaN or Inf
            if pd.isna(next_day_close_ret) or np.isinf(next_day_close_ret):
                continue

            all_sequences.append(sequence)
            all_targets.append([next_day_close_ret])
            symbol_samples += 1

        if symbol_samples > 0:
            symbols_used += 1
            total_samples += symbol_samples

    print(f"[PatchTST] Dataset built: {total_samples} daily samples from {symbols_used} symbols")

    if not all_sequences:
        empty_X = np.array([]).reshape(0, config.context_length, config.num_input_channels)
        empty_y = np.array([]).reshape(0, config.prediction_length)
        return DatasetResult(
            X=empty_X,
            y=empty_y,
            feature_scaler=StandardScaler(),
        )

    X = np.array(all_sequences)
    y = np.array(all_targets)

    # CRITICAL VERIFICATION: Dataset shape and channel count
    assert X.shape[2] == config.num_input_channels, \
        f"CRITICAL: Expected {config.num_input_channels} channels in X, got {X.shape[2]}"
    print(f"[PatchTST] VERIFY DATASET:")
    print(f"  X shape: {X.shape} (samples, context_length={config.context_length}, channels={config.num_input_channels})")
    print(f"  y shape: {y.shape} (samples, prediction_length={config.prediction_length})")
    print(f"  Expected channels: {config.feature_names}")
    print(f"  Target: next-day close_ret (channel {close_ret_idx})")
    
    # Verify no NaN/Inf in X
    x_nan_count = np.isnan(X).sum()
    x_inf_count = np.isinf(X).sum()
    y_nan_count = np.isnan(y).sum()
    y_inf_count = np.isinf(y).sum()
    if x_nan_count > 0:
        print(f"  ⚠️ CRITICAL: X has {x_nan_count} NaN values")
    if x_inf_count > 0:
        print(f"  ⚠️ CRITICAL: X has {x_inf_count} Inf values")
    if y_nan_count > 0:
        print(f"  ⚠️ CRITICAL: y has {y_nan_count} NaN values")
    if y_inf_count > 0:
        print(f"  ⚠️ CRITICAL: y has {y_inf_count} Inf values")

    # Fit feature scaler on input sequences
    original_shape = X.shape
    X_flat = X.reshape(-1, X.shape[-1])
    feature_scaler = StandardScaler()
    X_flat_scaled = feature_scaler.fit_transform(X_flat)
    X = X_flat_scaled.reshape(original_shape)

    # Log data statistics after scaling
    print(f"[PatchTST] Data statistics after scaling:")
    print(f"  X: mean={X.mean():.6f}, std={X.std():.6f}, min={X.min():.6f}, max={X.max():.6f}")
    print(f"  y: mean={y.mean():.6f}, std={y.std():.6f}, min={y.min():.6f}, max={y.max():.6f}")

    return DatasetResult(
        X=X,
        y=y,
        feature_scaler=feature_scaler,
    )

"""Dataset building for LSTM training."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from brain_api.core.features import compute_ohlcv_log_returns
from brain_api.core.lstm.config import LSTMConfig


@dataclass
class DatasetResult:
    """Result of dataset building for next-day return prediction."""

    X: np.ndarray  # Input sequences: (n_samples, seq_len, n_features)
    y: np.ndarray  # Targets: (n_samples, 1) - next-day returns
    feature_scaler: StandardScaler  # Scaler for input features


def build_dataset(
    prices: dict[str, pd.DataFrame],
    config: LSTMConfig,
) -> DatasetResult:
    """Build training dataset for next-day return prediction.

    Creates samples for each trading day:
    - Input: sequence_length days of OHLCV features
    - Target: next-day return (next_close - current_close) / current_close

    This enables iterative 5-day prediction for weekly forecasts, allowing
    computation of both weekly return (compounded) and volatility (std of daily).

    Args:
        prices: Dict of symbol -> OHLCV DataFrame with DatetimeIndex
        config: LSTM configuration

    Returns:
        DatasetResult with X, y (next-day returns), and feature_scaler
    """
    all_sequences = []
    all_targets = []

    print(f"[LSTM] Building dataset from {len(prices)} symbols...")
    symbols_used = 0
    total_samples = 0

    for _symbol, df in prices.items():
        # Skip if not enough data
        # Need at least sequence_length + 2 days (for features + target)
        if len(df) < config.sequence_length + 2:
            continue

        # Compute features using shared utility
        features_df = compute_ohlcv_log_returns(df, use_returns=config.use_returns)

        # Align original df with features (first row dropped when using returns)
        df_aligned = df.iloc[1:] if config.use_returns else df

        if len(features_df) < config.sequence_length + 1:
            continue

        symbol_samples = 0

        # For each day where we have enough history, create a training sample
        # Input: features from sequence_length days ending at day t
        # Target: return from day t to day t+1
        for t in range(config.sequence_length, len(features_df) - 1):
            # Extract input sequence (sequence_length days ending at day t)
            seq_start_idx = t - config.sequence_length
            seq_end_idx = t  # Exclusive, so features up to day t-1

            sequence = features_df.iloc[seq_start_idx:seq_end_idx].values

            if len(sequence) != config.sequence_length:
                continue

            # Compute target: next-day return (close-to-close)
            # day t close -> day t+1 close
            current_close = df_aligned["close"].iloc[t]
            next_close = df_aligned["close"].iloc[t + 1]

            if current_close == 0:
                continue

            next_day_return = (next_close - current_close) / current_close

            all_sequences.append(sequence)
            all_targets.append([next_day_return])
            symbol_samples += 1

        if symbol_samples > 0:
            symbols_used += 1
            total_samples += symbol_samples

    print(
        f"[LSTM] Dataset built: {total_samples} daily samples from {symbols_used} symbols"
    )

    if not all_sequences:
        # Return empty arrays if no data
        empty_X = np.array([]).reshape(0, config.sequence_length, config.input_size)
        empty_y = np.array([]).reshape(0, 1)
        return DatasetResult(
            X=empty_X,
            y=empty_y,
            feature_scaler=StandardScaler(),
        )

    X = np.array(all_sequences)
    y = np.array(all_targets)

    # Fit feature scaler on input sequences
    original_shape = X.shape
    X_flat = X.reshape(-1, X.shape[-1])
    feature_scaler = StandardScaler()
    X_flat_scaled = feature_scaler.fit_transform(X_flat)
    X = X_flat_scaled.reshape(original_shape)

    # Note: We do NOT scale the targets (daily returns).
    # Returns are already naturally bounded (typically -0.05 to +0.05) and
    # keeping them in original scale makes interpretation straightforward.

    return DatasetResult(
        X=X,
        y=y,
        feature_scaler=feature_scaler,
    )

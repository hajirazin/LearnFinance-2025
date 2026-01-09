"""Dataset building for LSTM training."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from brain_api.core.features import compute_ohlcv_log_returns
from brain_api.core.inference_utils import extract_trading_weeks
from brain_api.core.lstm.config import LSTMConfig


@dataclass
class DatasetResult:
    """Result of dataset building for weekly return prediction."""

    X: np.ndarray  # Input sequences: (n_samples, seq_len, n_features)
    y: np.ndarray  # Targets: (n_samples, 1) - weekly returns
    feature_scaler: StandardScaler  # Scaler for input features


def _compute_weekly_return(week_df: pd.DataFrame) -> float:
    """Compute weekly return from a trading week DataFrame.

    Args:
        week_df: DataFrame for a single trading week with OHLCV data

    Returns:
        Weekly return = (last_day_close - first_day_open) / first_day_open
    """
    first_day_open = week_df["open"].iloc[0]
    last_day_close = week_df["close"].iloc[-1]

    if first_day_open == 0:
        return 0.0

    return (last_day_close - first_day_open) / first_day_open


def build_dataset(
    prices: dict[str, pd.DataFrame],
    config: LSTMConfig,
) -> DatasetResult:
    """Build training dataset for weekly return prediction.

    Creates samples aligned to trading weeks:
    - Input: 60 trading days of features ending at week start
    - Target: weekly return (fri_close - mon_open) / mon_open

    This naturally handles holidays - a "week" is simply the first to last
    trading day of each ISO week.

    Args:
        prices: Dict of symbol -> OHLCV DataFrame with DatetimeIndex
        config: LSTM configuration

    Returns:
        DatasetResult with X, y (weekly returns), and feature_scaler
    """
    all_sequences = []
    all_targets = []

    print(f"[LSTM] Building dataset from {len(prices)} symbols...")
    symbols_used = 0
    total_weeks = 0

    for _symbol, df in prices.items():
        # Skip if not enough data
        if len(df) < config.sequence_length + 5:  # Need at least one week after lookback
            continue

        # Compute features using shared utility
        features_df = compute_ohlcv_log_returns(df, use_returns=config.use_returns)

        # Align original df with features (first row dropped when using returns)
        df_aligned = df.iloc[1:] if config.use_returns else df

        # Extract trading weeks from the aligned data
        weeks = extract_trading_weeks(df_aligned, min_days=config.min_week_days)

        if len(weeks) < 2:  # Need at least 2 weeks (1 for target)
            continue

        symbol_samples = 0

        # For each week (except the last), create a training sample
        # Input: features from sequence_length days ending at week start
        # Target: that week's return
        for i in range(len(weeks) - 1):
            week = weeks[i + 1]  # Target week (we predict the NEXT week)
            week_start = week.index[0]

            # Find the position of week_start in features_df
            try:
                week_start_idx = features_df.index.get_loc(week_start)
            except KeyError:
                continue

            # Check if we have enough history
            if week_start_idx < config.sequence_length:
                continue

            # Extract input sequence (sequence_length days ending just before week start)
            seq_start_idx = week_start_idx - config.sequence_length
            seq_end_idx = week_start_idx  # Exclusive

            sequence = features_df.iloc[seq_start_idx:seq_end_idx].values

            if len(sequence) != config.sequence_length:
                continue

            # Compute target: weekly return for the target week
            weekly_return = _compute_weekly_return(week)

            all_sequences.append(sequence)
            all_targets.append([weekly_return])
            symbol_samples += 1

        if symbol_samples > 0:
            symbols_used += 1
            total_weeks += symbol_samples

    print(f"[LSTM] Dataset built: {total_weeks} weekly samples from {symbols_used} symbols")

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

    # Note: We do NOT scale the targets (weekly returns).
    # Returns are already naturally bounded (typically -0.1 to +0.1) and
    # keeping them in original scale makes interpretation straightforward.

    return DatasetResult(
        X=X,
        y=y,
        feature_scaler=feature_scaler,
    )

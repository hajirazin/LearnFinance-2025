"""Dataset building for LSTM training.

Builds week-aligned samples for direct 5-day close-return prediction.
Each sample is anchored at the last trading day of a week. The input
is a sequence of OHLCV log returns ending at the anchor day (inclusive),
and the target is the next 5 close-to-close log returns.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from brain_api.core.features import compute_ohlcv_log_returns
from brain_api.core.lstm.config import LSTMConfig


@dataclass
class DatasetResult:
    """Result of dataset building for direct 5-day close-return prediction."""

    X: np.ndarray  # Input sequences: (n_samples, seq_len, n_features)
    y: np.ndarray  # Targets: (n_samples, 5) - next 5 close log returns
    feature_scaler: StandardScaler  # Scaler for input features


def build_dataset(
    prices: dict[str, pd.DataFrame],
    config: LSTMConfig,
) -> DatasetResult:
    """Build training dataset for direct 5-day close-return prediction.

    Creates week-aligned samples:
    - Anchors at the last trading day of each week (detected by calendar gap >= 2 days)
    - Input: sequence_length days of OHLCV log returns ending at anchor (inclusive)
    - Target: next 5 close-to-close log returns after the anchor

    The input includes the anchor day's return so the model knows the close
    price at the anchor. Targets start from the next trading day.

    No holiday logic is needed: targets are always the next 5 rows in
    features_df, which are trading days by definition. Holidays don't
    appear in the data.

    Args:
        prices: Dict of symbol -> OHLCV DataFrame with DatetimeIndex
        config: LSTM configuration

    Returns:
        DatasetResult with X, y (5 close log returns per sample), and feature_scaler
    """
    all_sequences = []
    all_targets = []

    print(f"[LSTM] Building dataset from {len(prices)} symbols...")
    symbols_used = 0
    total_samples = 0

    for _symbol, df in prices.items():
        # Need at least sequence_length + 7 days (for features + 5 targets + returns shift)
        if len(df) < config.sequence_length + 7:
            continue

        # Compute features using shared utility (OHLCV log returns, 5 features)
        features_df = compute_ohlcv_log_returns(df, use_returns=config.use_returns)

        if len(features_df) < config.sequence_length + 5:
            continue

        # Detect week-end anchors: last trading day of each week
        # A week-end is any day followed by a gap of >= 2 calendar days
        # (weekends have 2-day gap, holiday weekends have 3+)
        dates = features_df.index
        n_dates = len(dates)

        # Compute gaps between consecutive trading days
        gaps = pd.Series(
            [(dates[i + 1] - dates[i]).days for i in range(n_dates - 1)],
            index=range(n_dates - 1),
        )
        week_ends = [i for i in range(len(gaps)) if gaps.iloc[i] >= 2]

        symbol_samples = 0

        for t in week_ends:
            # Skip if not enough history for input sequence
            if t < config.sequence_length - 1:
                continue

            # Skip if not enough future data for 5 targets
            if t + 5 >= n_dates:
                continue

            # Input: seq_len days ending at anchor (inclusive)
            # Bug #2 fix: include anchor day's return so model knows close at t
            seq_start = t - config.sequence_length + 1
            seq_end = t + 1  # exclusive, so includes position t
            sequence = features_df.iloc[seq_start:seq_end].values

            if len(sequence) != config.sequence_length:
                continue

            # Target: next 5 close log returns after anchor
            # close_ret is the 4th column (index 3) in OHLCV log returns
            target = features_df.iloc[t + 1 : t + 6]["close_ret"].values

            if len(target) != 5:
                continue

            # Skip if any NaN or Inf in target
            if np.any(np.isnan(target)) or np.any(np.isinf(target)):
                continue

            all_sequences.append(sequence)
            all_targets.append(target)
            symbol_samples += 1

        if symbol_samples > 0:
            symbols_used += 1
            total_samples += symbol_samples

    print(
        f"[LSTM] Dataset built: {total_samples} week-aligned samples "
        f"from {symbols_used} symbols"
    )

    if not all_sequences:
        # Return empty arrays if no data
        empty_X = np.array([]).reshape(0, config.sequence_length, config.input_size)
        empty_y = np.array([]).reshape(0, 5)
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

    # Note: We do NOT scale the targets (close log returns).
    # Returns are already naturally bounded (typically -0.05 to +0.05) and
    # keeping them in original scale means the model's linear layer outputs
    # raw log returns directly. NO inverse transform needed at inference.

    return DatasetResult(
        X=X,
        y=y,
        feature_scaler=feature_scaler,
    )

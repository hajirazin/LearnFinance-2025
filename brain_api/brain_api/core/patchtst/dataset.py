"""Dataset building for PatchTST training.

Builds week-aligned samples for direct 5-day multi-task prediction.
Each sample is anchored at the last trading day of a week. The input
is a sequence of OHLCV log returns ending at the anchor day (inclusive),
and the target is all 5 OHLCV log returns for the next 5 trading days.

Multi-task: targets include ALL 5 channels (open_ret, high_ret, low_ret,
close_ret, volume_ret) so the shared Transformer weights learn from all
channels simultaneously (data augmentation effect).

Targets and inputs are UNSCALED -- RevIN inside PatchTST handles
per-channel per-sample normalization internally.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from brain_api.core.patchtst.config import PatchTSTConfig


@dataclass
class DatasetResult:
    """Result of dataset building for direct 5-day multi-task prediction.

    X: Input sequences of OHLCV log returns (UNSCALED -- RevIN normalizes internally).
    y: Targets of ALL 5 OHLCV channels for next 5 trading days (UNSCALED).
    feature_scaler: Fitted for diagnostics only (data drift monitoring). NOT used for model normalization.
    """

    X: (
        np.ndarray
    )  # Input sequences: (n_samples, context_length, 5) -- OHLCV log returns
    y: np.ndarray  # Targets: (n_samples, 5, 5) -- (5 days, 5 channels) UNSCALED
    feature_scaler: StandardScaler  # Diagnostic only -- NOT applied to X or y


def build_dataset(
    aligned_features: dict[str, pd.DataFrame],
    prices: dict[str, pd.DataFrame],
    config: PatchTSTConfig,
) -> DatasetResult:
    """Build training dataset for direct 5-day multi-task prediction.

    Creates week-aligned samples:
    - Anchors at the last trading day of each week (detected by calendar gap >= 2 days)
    - Input: context_length days of OHLCV log returns ending at anchor (inclusive)
    - Target: ALL 5 OHLCV channels for the next 5 trading days

    The aligned_features DataFrames may contain up to 12 channels (including
    news/fundamentals). Only the 5 OHLCV columns are extracted for model input
    and targets.

    X and y are UNSCALED raw log returns. PatchTST's internal RevIN handles
    per-channel per-sample normalization. A StandardScaler is still fitted on X
    for diagnostics (data drift monitoring) but is NOT applied.

    No holiday logic is needed: targets are always the next 5 rows in
    the DataFrame, which are trading days by definition.

    Args:
        aligned_features: Dict of symbol -> aligned multi-channel DataFrame
                         (output from align_multivariate_data, may have 12 cols)
        prices: Dict of symbol -> raw OHLCV DataFrame (not used, kept for interface compatibility)
        config: PatchTST configuration

    Returns:
        DatasetResult with X (UNSCALED), y (UNSCALED), and diagnostic feature_scaler
    """
    ohlcv_cols = ["open_ret", "high_ret", "low_ret", "close_ret", "volume_ret"]

    all_sequences = []
    all_targets = []

    print(f"[PatchTST] Building dataset from {len(aligned_features)} symbols...")
    symbols_used = 0
    total_samples = 0

    for _symbol, features_df in aligned_features.items():
        # Extract OHLCV columns only (5 channels)
        missing_cols = [c for c in ohlcv_cols if c not in features_df.columns]
        if missing_cols:
            print(f"[PatchTST] Skipping {_symbol}: missing columns {missing_cols}")
            continue

        ohlcv_df = features_df[ohlcv_cols]

        # Need at least context_length + 5 rows
        if len(ohlcv_df) < config.context_length + 5:
            continue

        # Detect week-end anchors: last trading day of each week
        # A week-end is any day followed by a gap of >= 2 calendar days
        # (weekends have 2-day gap, holiday weekends have 3+)
        dates = ohlcv_df.index
        n_dates = len(dates)

        gaps = pd.Series(
            [(dates[i + 1] - dates[i]).days for i in range(n_dates - 1)],
            index=range(n_dates - 1),
        )
        week_ends = [i for i in range(len(gaps)) if gaps.iloc[i] >= 2]

        symbol_samples = 0

        for t in week_ends:
            # Skip if not enough history for input sequence
            if t < config.context_length - 1:
                continue

            # Skip if not enough future data for 5-day target
            if t + 5 >= n_dates:
                continue

            # Input: context_length days ending at anchor (inclusive)
            seq_start = t - config.context_length + 1
            seq_end = t + 1  # exclusive, so includes position t
            sequence = ohlcv_df.iloc[seq_start:seq_end].values  # (context_length, 5)

            if len(sequence) != config.context_length:
                continue

            # Target: ALL 5 OHLCV channels for next 5 trading days
            target = ohlcv_df.iloc[t + 1 : t + 6].values  # (5, 5) = 5 days x 5 channels

            if target.shape != (5, 5):
                continue

            # Skip if any NaN or Inf in sequence or target
            if np.any(np.isnan(sequence)) or np.any(np.isinf(sequence)):
                continue
            if np.any(np.isnan(target)) or np.any(np.isinf(target)):
                continue

            all_sequences.append(sequence)
            all_targets.append(target)
            symbol_samples += 1

        if symbol_samples > 0:
            symbols_used += 1
            total_samples += symbol_samples

    print(
        f"[PatchTST] Dataset built: {total_samples} week-aligned samples "
        f"from {symbols_used} symbols"
    )

    if not all_sequences:
        empty_X = np.array([]).reshape(0, config.context_length, 5)
        empty_y = np.array([]).reshape(0, 5, 5)
        return DatasetResult(
            X=empty_X,
            y=empty_y,
            feature_scaler=StandardScaler(),
        )

    X = np.array(all_sequences)  # (n_samples, context_length, 5)
    y = np.array(all_targets)  # (n_samples, 5, 5)

    # CRITICAL VERIFICATION: Dataset shape and channel count
    assert X.shape[2] == 5, f"CRITICAL: Expected 5 channels in X, got {X.shape[2]}"
    assert y.shape[1:] == (5, 5), f"CRITICAL: Expected y shape (n, 5, 5), got {y.shape}"

    print("[PatchTST] VERIFY DATASET:")
    print(
        f"  X shape: {X.shape} (samples, context_length={config.context_length}, channels=5)"
    )
    print(f"  y shape: {y.shape} (samples, 5 days, 5 channels)")
    print(f"  Channels: {ohlcv_cols}")
    print("  Targets: ALL 5 channels for next 5 trading days (UNSCALED)")

    # Verify no NaN/Inf
    x_nan_count = np.isnan(X).sum()
    x_inf_count = np.isinf(X).sum()
    y_nan_count = np.isnan(y).sum()
    y_inf_count = np.isinf(y).sum()
    if x_nan_count > 0:
        print(f"  WARNING: X has {x_nan_count} NaN values")
    if x_inf_count > 0:
        print(f"  WARNING: X has {x_inf_count} Inf values")
    if y_nan_count > 0:
        print(f"  WARNING: y has {y_nan_count} NaN values")
    if y_inf_count > 0:
        print(f"  WARNING: y has {y_inf_count} Inf values")

    # Fit scaler for diagnostics ONLY (data drift monitoring)
    # DO NOT apply to X or y -- RevIN handles normalization internally
    feature_scaler = StandardScaler()
    feature_scaler.fit(X.reshape(-1, 5))  # fit only, don't transform

    # Log data statistics (raw, unscaled)
    print("[PatchTST] Data statistics (raw, unscaled -- RevIN normalizes internally):")
    print(
        f"  X: mean={X.mean():.6f}, std={X.std():.6f}, min={X.min():.6f}, max={X.max():.6f}"
    )
    print(
        f"  y: mean={y.mean():.6f}, std={y.std():.6f}, min={y.min():.6f}, max={y.max():.6f}"
    )

    return DatasetResult(
        X=X,
        y=y,
        feature_scaler=feature_scaler,
    )

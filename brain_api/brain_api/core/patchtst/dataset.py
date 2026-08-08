"""Dataset building for PatchTST training.

Builds week-aligned samples for direct 5-day OHLCV targets (loss uses close
only at train time). Each sample is anchored at the **last trading day of an
ISO week** with at least ``config.min_week_days`` sessions.

The input is a sequence of OHLCV log returns ending at the anchor day
(inclusive), and the target is all 5 OHLCV log returns for the next 5 trading
days (targets stored for all channels; training optimizes close only).

Targets and inputs are UNSCALED -- RevIN inside PatchTST handles
per-channel per-sample normalization internally.
"""

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from brain_api.core.patchtst.config import PatchTSTConfig


@dataclass
class DatasetResult:
    """Result of dataset building for direct 5-day multi-channel targets.

    X: Input sequences of OHLCV log returns (UNSCALED -- RevIN normalizes internally).
    y: Targets of ALL 5 OHLCV channels for next 5 trading days (UNSCALED).
    feature_scaler: Fitted for diagnostics only (data drift monitoring). NOT used for model normalization.
    anchor_dates: Per-sample week-end anchor dates (chronological split).
    """

    X: np.ndarray  # (n_samples, context_length, 5)
    y: np.ndarray  # (n_samples, 5, 5)
    feature_scaler: StandardScaler
    anchor_dates: np.ndarray  # dtype=object, date objects length n_samples


def _week_end_anchors(dates: pd.DatetimeIndex, min_week_days: int) -> list[int]:
    """Return positional indices of ISO-week last trading days.

    Uses pandas ``W`` periods (week ending Sunday). Mid-week holidays no longer
    create false anchors (unlike gap>=2 heuristics).
    """
    if len(dates) == 0:
        return []

    periods = dates.to_period("W")
    anchors: list[int] = []
    # Group consecutive positions by period label
    i = 0
    n = len(dates)
    while i < n:
        period = periods[i]
        j = i + 1
        while j < n and periods[j] == period:
            j += 1
        week_len = j - i
        if week_len >= min_week_days:
            anchors.append(j - 1)  # last trading day in this ISO week
        i = j
    return anchors


def build_dataset(
    aligned_features: dict[str, pd.DataFrame],
    prices: dict[str, pd.DataFrame],
    config: PatchTSTConfig,
) -> DatasetResult:
    """Build training dataset for direct 5-day OHLCV targets.

    Creates week-aligned samples:
    - Anchors at the last trading day of each ISO week with >= min_week_days
    - Input: context_length days of OHLCV log returns ending at anchor (inclusive)
    - Target: ALL 5 OHLCV channels for the next 5 trading days

    X and y are UNSCALED raw log returns. Samples are returned sorted by
    ``anchor_dates`` ascending for chronological train/val splits.
    """
    ohlcv_cols = ["open_ret", "high_ret", "low_ret", "close_ret", "volume_ret"]

    all_sequences = []
    all_targets = []
    all_anchors: list[date] = []

    print(f"[PatchTST] Building dataset from {len(aligned_features)} symbols...")
    symbols_used = 0
    total_samples = 0

    for _symbol, features_df in aligned_features.items():
        missing_cols = [c for c in ohlcv_cols if c not in features_df.columns]
        if missing_cols:
            print(f"[PatchTST] Skipping {_symbol}: missing columns {missing_cols}")
            continue

        ohlcv_df = features_df[ohlcv_cols]

        if len(ohlcv_df) < config.context_length + 5:
            continue

        dates = ohlcv_df.index
        n_dates = len(dates)
        week_ends = _week_end_anchors(dates, config.min_week_days)

        symbol_samples = 0

        for t in week_ends:
            if t < config.context_length - 1:
                continue
            if t + 5 >= n_dates:
                continue

            seq_start = t - config.context_length + 1
            seq_end = t + 1
            sequence = ohlcv_df.iloc[seq_start:seq_end].values

            if len(sequence) != config.context_length:
                continue

            target = ohlcv_df.iloc[t + 1 : t + 6].values

            if target.shape != (5, 5):
                continue

            if np.any(np.isnan(sequence)) or np.any(np.isinf(sequence)):
                continue
            if np.any(np.isnan(target)) or np.any(np.isinf(target)):
                continue

            anchor_ts = dates[t]
            anchor_d = (
                anchor_ts.date()
                if hasattr(anchor_ts, "date")
                else pd.Timestamp(anchor_ts).date()
            )

            all_sequences.append(sequence)
            all_targets.append(target)
            all_anchors.append(anchor_d)
            symbol_samples += 1

        if symbol_samples > 0:
            symbols_used += 1
            total_samples += symbol_samples

    print(
        f"[PatchTST] Dataset built: {total_samples} week-aligned samples "
        f"from {symbols_used} symbols"
    )

    if not all_sequences:
        empty_X = np.empty((0, config.context_length, 5), dtype=np.float32)
        empty_y = np.empty((0, 5, 5), dtype=np.float32)
        return DatasetResult(
            X=empty_X,
            y=empty_y,
            feature_scaler=StandardScaler(),
            anchor_dates=np.array([], dtype=object),
        )

    X = np.array(all_sequences, dtype=np.float32)
    y = np.array(all_targets, dtype=np.float32)
    anchor_dates = np.array(all_anchors, dtype=object)
    del all_sequences, all_targets

    # Chronological order for time-based train/val (Phase D)
    order = np.argsort(anchor_dates)
    X = X[order]
    y = y[order]
    anchor_dates = anchor_dates[order]

    assert X.shape[2] == 5, f"CRITICAL: Expected 5 channels in X, got {X.shape[2]}"
    assert y.shape[1:] == (5, 5), f"CRITICAL: Expected y shape (n, 5, 5), got {y.shape}"

    print("[PatchTST] VERIFY DATASET:")
    print(
        f"  X shape: {X.shape} (samples, context_length={config.context_length}, channels=5)"
    )
    print(f"  y shape: {y.shape} (samples, 5 days, 5 channels)")
    print(f"  Channels: {ohlcv_cols}")
    print(
        f"  Anchors: ISO-week last session, min_week_days={config.min_week_days}; "
        f"sorted {anchor_dates[0]} .. {anchor_dates[-1]}"
    )
    print(
        "  Targets: ALL 5 channels for next 5 trading days (UNSCALED); loss uses close only"
    )

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

    feature_scaler = StandardScaler()
    feature_scaler.fit(X.reshape(-1, 5))

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
        anchor_dates=anchor_dates,
    )

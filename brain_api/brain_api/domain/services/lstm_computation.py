"""LSTM computation domain service.

Pure functions for version computation, week boundary calculation,
and feature building. These have no external dependencies except
pandas/numpy and domain entities.
"""

import hashlib
import json
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from brain_api.domain.entities.lstm import (
    InferenceFeatures,
    LSTMConfig,
    SymbolPrediction,
    WeekBoundaries,
)


def compute_version(
    start_date: date,
    end_date: date,
    symbols: list[str],
    config: LSTMConfig,
) -> str:
    """Compute a deterministic version string for a training run.

    The version is a hash of (window, symbols, config) so that reruns with
    the same inputs produce the same version (idempotent training).

    Args:
        start_date: Training data start date
        end_date: Training data end date
        symbols: List of symbols in training set
        config: LSTM configuration

    Returns:
        Version string in format 'v{date_prefix}-{hash_suffix}'
    """
    # Create a canonical representation
    canonical = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "symbols": sorted(symbols),  # Sort for determinism
        "config": config.to_dict(),
    }
    canonical_json = json.dumps(canonical, sort_keys=True)

    # Hash it
    hash_digest = hashlib.sha256(canonical_json.encode()).hexdigest()[:12]

    # Include end_date in version for human readability
    date_prefix = end_date.strftime("%Y-%m-%d")

    return f"v{date_prefix}-{hash_digest}"


def compute_week_boundaries_simple(as_of_date: date) -> WeekBoundaries:
    """Compute week boundaries without exchange calendar (simple version).

    This is a pure computation without external dependencies.
    For holiday-aware boundaries, use the infrastructure layer's
    version with exchange-calendars.

    Args:
        as_of_date: Reference date within the target week

    Returns:
        WeekBoundaries with calendar-based dates
    """
    # Get the Monday of as_of_date's week
    days_since_monday = as_of_date.weekday()  # 0 = Monday
    calendar_monday = as_of_date - timedelta(days=days_since_monday)
    calendar_friday = calendar_monday + timedelta(days=4)

    return WeekBoundaries(
        target_week_start=calendar_monday,
        target_week_end=calendar_friday,
        calendar_monday=calendar_monday,
        calendar_friday=calendar_friday,
    )


def extract_trading_weeks(df: pd.DataFrame, min_days: int = 3) -> list[pd.DataFrame]:
    """Extract trading weeks from a price DataFrame.

    Groups data by ISO week and filters out weeks with too few trading days.

    Args:
        df: DataFrame with DatetimeIndex containing OHLCV data
        min_days: Minimum trading days required in a week

    Returns:
        List of DataFrames, one per valid week
    """
    if df.empty:
        return []

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    # Group by ISO year-week
    df_copy = df.copy()
    df_copy["iso_year"] = df_copy.index.isocalendar().year
    df_copy["iso_week"] = df_copy.index.isocalendar().week

    weeks = []
    for (year, week), group in df_copy.groupby(["iso_year", "iso_week"]):
        # Drop the grouping columns
        week_df = group.drop(columns=["iso_year", "iso_week"])
        if len(week_df) >= min_days:
            weeks.append(week_df)

    return weeks


def compute_weekly_return(week_df: pd.DataFrame) -> float | None:
    """Compute weekly return from a week's price data.

    Weekly return = (Friday close - Monday open) / Monday open

    Args:
        week_df: DataFrame with OHLCV data for one week

    Returns:
        Weekly return as decimal (e.g., 0.02 for +2%), or None if data missing
    """
    if week_df.empty:
        return None

    # Get Monday open (first row) and Friday close (last row)
    # Column names are lowercase
    try:
        monday_open = week_df.iloc[0]["open"]
        friday_close = week_df.iloc[-1]["close"]

        if pd.isna(monday_open) or pd.isna(friday_close) or monday_open == 0:
            return None

        return (friday_close - monday_open) / monday_open
    except (KeyError, IndexError):
        return None


def build_inference_features_from_data(
    symbol: str,
    prices_df: pd.DataFrame,
    config: LSTMConfig,
    cutoff_date: date,
) -> InferenceFeatures:
    """Build inference features from pre-loaded price data.

    This is a pure computation - it doesn't fetch data itself.

    Args:
        symbol: Stock ticker symbol
        prices_df: DataFrame with OHLCV columns
        config: LSTM configuration
        cutoff_date: Last date to use (exclusive of target week)

    Returns:
        InferenceFeatures with prepared input data
    """
    if prices_df is None or prices_df.empty:
        return InferenceFeatures(
            symbol=symbol,
            features=None,
            has_enough_history=False,
            history_days_used=0,
            data_end_date=None,
        )

    # Ensure DatetimeIndex
    if not isinstance(prices_df.index, pd.DatetimeIndex):
        prices_df = prices_df.copy()
        prices_df.index = pd.to_datetime(prices_df.index)

    # Filter to before cutoff date
    mask = prices_df.index.date < cutoff_date
    df = prices_df.loc[mask].copy()

    if len(df) < config.sequence_length + 1:
        return InferenceFeatures(
            symbol=symbol,
            features=None,
            has_enough_history=False,
            history_days_used=len(df),
            data_end_date=df.index[-1].date() if len(df) > 0 else None,
        )

    # Use last sequence_length + 1 days (need +1 for log returns)
    df = df.tail(config.sequence_length + 1)

    # Compute log returns
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    price_cols = ["open", "high", "low", "close"]

    # Ensure columns exist (case-insensitive matching)
    col_map = {}
    for target in ohlcv_cols:
        for col in df.columns:
            if col.lower() == target:
                col_map[target] = col
                break

    if len(col_map) < 5:
        return InferenceFeatures(
            symbol=symbol,
            features=None,
            has_enough_history=False,
            history_days_used=len(df),
            data_end_date=df.index[-1].date() if len(df) > 0 else None,
        )

    # Build features array
    features = []
    for col in ohlcv_cols:
        src_col = col_map[col]
        if col in price_cols:
            # Log returns for price columns
            values = np.log(df[src_col] / df[src_col].shift(1)).dropna().values
        else:
            # Log volume (not returns)
            values = np.log(df[src_col] + 1).iloc[1:].values

        features.append(values)

    # Stack: (seq_len, n_features)
    features_array = np.column_stack(features)

    if len(features_array) < config.sequence_length:
        return InferenceFeatures(
            symbol=symbol,
            features=None,
            has_enough_history=False,
            history_days_used=len(features_array),
            data_end_date=df.index[-1].date(),
        )

    return InferenceFeatures(
        symbol=symbol,
        features=features_array[-config.sequence_length :],
        has_enough_history=True,
        history_days_used=len(features_array),
        data_end_date=df.index[-1].date(),
    )


def classify_direction(predicted_return_pct: float | None) -> str:
    """Classify prediction direction.

    Args:
        predicted_return_pct: Predicted return as percentage

    Returns:
        "UP", "DOWN", or "FLAT"
    """
    if predicted_return_pct is None:
        return "FLAT"
    if predicted_return_pct > 0.5:  # > 0.5% threshold
        return "UP"
    elif predicted_return_pct < -0.5:  # < -0.5% threshold
        return "DOWN"
    else:
        return "FLAT"


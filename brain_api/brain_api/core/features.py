"""Shared feature engineering utilities for all model types."""

import numpy as np
import pandas as pd

from brain_api.core.prices import repair_ohlc_envelope


def _log_return_positive(series: pd.Series) -> pd.Series:
    """Log return where both current and previous values are strictly positive.

    Non-positive or non-finite bars (today or previous) become NaN so
    downstream dataset builders skip those windows. Never substitutes a
    sentinel (e.g. ``0 -> 1``) and never emits Inf from ``log(0/x)``.
    """
    prev = series.shift(1)
    curr_v = series.to_numpy(dtype=np.float64, copy=False)
    prev_v = prev.to_numpy(dtype=np.float64, copy=False)
    valid = np.isfinite(curr_v) & np.isfinite(prev_v) & (curr_v > 0.0) & (prev_v > 0.0)
    out = np.full(len(series), np.nan, dtype=np.float64)
    if valid.any():
        out[valid] = np.log(curr_v[valid] / prev_v[valid])
    return pd.Series(out, index=series.index, dtype="float64")


def compute_ohlcv_log_returns(
    df: pd.DataFrame, use_returns: bool = True
) -> pd.DataFrame:
    """Compute OHLCV log returns from price DataFrame.

    Transforms raw OHLCV prices into log returns for improved stationarity
    in time series models.

    Args:
        df: DataFrame with columns: open, high, low, close, volume
            and DatetimeIndex
        use_returns: If True, compute log returns. If False, just rename columns.

    Returns:
        DataFrame with columns: open_ret, high_ret, low_ret, close_ret, volume_ret
        First row is dropped when use_returns=True (NaN from shift).
        Non-positive OHLCV (today or previous bar) yields NaN on that channel —
        never Inf and never a silent zero-fill (AGENTS.md: no silent fallbacks).
        Downstream dataset builders skip NaN/Inf samples.
    """
    repaired = repair_ohlc_envelope(df)
    if use_returns:
        features_df = pd.DataFrame(
            {
                "open_ret": _log_return_positive(repaired["open"]),
                "high_ret": _log_return_positive(repaired["high"]),
                "low_ret": _log_return_positive(repaired["low"]),
                "close_ret": _log_return_positive(repaired["close"]),
                "volume_ret": _log_return_positive(repaired["volume"]),
            },
            index=repaired.index,
        )
        # Drop first row (NaN from shift)
        features_df = features_df.iloc[1:]
    else:
        features_df = repaired[["open", "high", "low", "close", "volume"]].copy()
        features_df.columns = [
            "open_ret",
            "high_ret",
            "low_ret",
            "close_ret",
            "volume_ret",
        ]

    return features_df


def compute_weekly_return(
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

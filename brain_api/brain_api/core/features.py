"""Shared feature engineering utilities for all model types."""

import numpy as np
import pandas as pd


def compute_ohlcv_log_returns(df: pd.DataFrame, use_returns: bool = True) -> pd.DataFrame:
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
        Infinities replaced with 0, NaN filled with 0.
    """
    if use_returns:
        features_df = pd.DataFrame(
            {
                "open_ret": np.log(df["open"] / df["open"].shift(1)),
                "high_ret": np.log(df["high"] / df["high"].shift(1)),
                "low_ret": np.log(df["low"] / df["low"].shift(1)),
                "close_ret": np.log(df["close"] / df["close"].shift(1)),
                "volume_ret": np.log(
                    df["volume"] / df["volume"].shift(1).replace(0, 1)
                ),
            },
            index=df.index,
        )
        # Drop first row (NaN from shift)
        features_df = features_df.iloc[1:]
    else:
        features_df = df[["open", "high", "low", "close", "volume"]].copy()
        features_df.columns = ["open_ret", "high_ret", "low_ret", "close_ret", "volume_ret"]

    # Replace infinities with 0 and fill NaN
    features_df = features_df.replace([np.inf, -np.inf], 0).fillna(0)

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


"""Data loading utilities for PatchTST training and inference."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from brain_api.core.features import compute_ohlcv_log_returns
from brain_api.core.fundamentals import load_historical_fundamentals_from_cache
from brain_api.core.patchtst.config import PatchTSTConfig

# Alias for backward compatibility
load_historical_fundamentals = load_historical_fundamentals_from_cache


def load_historical_news_sentiment(
    symbols: list[str],
    start_date: date,
    end_date: date,
    parquet_path: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Load historical news sentiment from parquet file.

    Args:
        symbols: List of ticker symbols
        start_date: Start of data window
        end_date: End of data window
        parquet_path: Path to daily_sentiment.parquet (defaults to project data/)

    Returns:
        Dict mapping symbol -> DataFrame with 'sentiment_score' column and DatetimeIndex
    """
    if parquet_path is None:
        # Default path: project_root/data/output/daily_sentiment.parquet
        parquet_path = (
            Path(__file__).parent.parent.parent.parent
            / "data"
            / "output"
            / "daily_sentiment.parquet"
        )

    sentiment: dict[str, pd.DataFrame] = {}

    if not parquet_path.exists():
        print(f"[PatchTST] Warning: News sentiment parquet not found at {parquet_path}")
        return sentiment

    try:
        df = pd.read_parquet(parquet_path)
        # Convert date column to string for filtering
        df["date"] = pd.to_datetime(df["date"]).dt.date

        for symbol in symbols:
            symbol_df = df[
                (df["symbol"] == symbol)
                & (df["date"] >= start_date)
                & (df["date"] <= end_date)
            ][["date", "sentiment_score"]].copy()

            if len(symbol_df) > 0:
                symbol_df["date"] = pd.to_datetime(symbol_df["date"])
                symbol_df = symbol_df.set_index("date").sort_index()
                sentiment[symbol] = symbol_df

    except Exception as e:
        print(f"[PatchTST] Error loading news sentiment: {e}")

    return sentiment


def align_multivariate_data(
    prices: dict[str, pd.DataFrame],
    config: PatchTSTConfig,
) -> dict[str, pd.DataFrame]:
    """Align OHLCV data into feature channels for PatchTST (5-channel OHLCV only).

    Computes OHLCV log returns and filters to config.feature_names (5 channels:
    open_ret, high_ret, low_ret, close_ret, volume_ret). The model uses no signals.

    Args:
        prices: Dict of symbol -> OHLCV DataFrame with DatetimeIndex
        config: PatchTST configuration (num_input_channels=5, feature_names=OHLCV only)

    Returns:
        Dict of symbol -> aligned DataFrame with config.num_input_channels columns
    """
    aligned: dict[str, pd.DataFrame] = {}

    for symbol, price_df in prices.items():
        if len(price_df) < config.context_length + 5:
            continue

        # OHLCV log returns only (5 channels)
        features_df = compute_ohlcv_log_returns(
            price_df, use_returns=config.use_returns
        )

        # Ensure column order matches config.feature_names
        features_df = features_df[config.feature_names]

        # CRITICAL VERIFICATION: Channel count
        assert len(features_df.columns) == config.num_input_channels, (
            f"CRITICAL: Expected {config.num_input_channels} channels, got {len(features_df.columns)}"
        )

        # Quick data quality check (no heavy stats computation)
        nan_count = features_df.isna().sum().sum()
        inf_count = np.isinf(features_df.select_dtypes(include=[np.number])).sum().sum()

        # Only log warnings for problematic symbols
        if nan_count > 0:
            print(f"[PatchTST] WARNING: {symbol} has {nan_count} NaN values")
        if inf_count > 0:
            print(f"[PatchTST] WARNING: {symbol} has {inf_count} Inf values")

        if len(features_df) >= config.context_length:
            aligned[symbol] = features_df

    # Summary log at the end (not per-symbol)
    print(
        f"[PatchTST] Aligned {len(aligned)} symbols with {config.num_input_channels} channels each"
    )

    return aligned

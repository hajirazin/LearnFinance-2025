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
    news_sentiment: dict[str, pd.DataFrame],
    fundamentals: dict[str, pd.DataFrame],
    config: PatchTSTConfig,
) -> dict[str, pd.DataFrame]:
    """Align OHLCV, news sentiment, and fundamentals into multi-channel features.

    Creates a unified DataFrame per symbol with all 11 feature channels:
    - OHLCV log returns (5 channels)
    - News sentiment (1 channel) - forward-filled for missing days
    - Fundamentals (5 channels) - forward-filled quarterly data

    Args:
        prices: Dict of symbol -> OHLCV DataFrame with DatetimeIndex
        news_sentiment: Dict of symbol -> sentiment DataFrame
        fundamentals: Dict of symbol -> fundamentals DataFrame
        config: PatchTST configuration

    Returns:
        Dict of symbol -> aligned multi-channel DataFrame
    """
    aligned: dict[str, pd.DataFrame] = {}

    for symbol, price_df in prices.items():
        if len(price_df) < config.context_length + 5:
            continue

        # Start with OHLCV features using shared utility
        features_df = compute_ohlcv_log_returns(
            price_df, use_returns=config.use_returns
        )

        # Add news sentiment (forward-fill missing days)
        if symbol in news_sentiment:
            sentiment_df = news_sentiment[symbol]
            # Reindex to match price dates and forward-fill
            sentiment_aligned = sentiment_df.reindex(features_df.index, method="ffill")
            features_df["news_sentiment"] = sentiment_aligned["sentiment_score"].fillna(
                0.0
            )
        else:
            features_df["news_sentiment"] = 0.0  # Neutral if no news data

        # Add fundamentals (forward-fill quarterly data)
        fundamental_cols = [
            "gross_margin",
            "operating_margin",
            "net_margin",
            "current_ratio",
            "debt_to_equity",
        ]
        if symbol in fundamentals:
            fund_df = fundamentals[symbol]
            # Reindex to match price dates and forward-fill
            fund_aligned = fund_df.reindex(features_df.index, method="ffill")

            # Vectorized calculation of days since last fundamental update
            # Use searchsorted to find the position of each date in the sorted fund_df index
            fund_dates = fund_df.index.values
            if len(fund_dates) > 0:
                # searchsorted returns position where date would be inserted
                # side='right' means we get the index after the last <= date
                positions = np.searchsorted(
                    fund_dates, features_df.index.values, side="right"
                )
                # Clip to valid indices (position - 1 gives us the last date <= current)
                valid_positions = np.clip(positions - 1, 0, len(fund_dates) - 1)
                # Get the last update dates
                last_updates = fund_dates[valid_positions]
                # Calculate days old (vectorized)
                days_old = (
                    (features_df.index.values - last_updates)
                    .astype("timedelta64[D]")
                    .astype(float)
                )
                # Handle cases where position is 0 and date is before first fundamental
                days_old[positions == 0] = 999.0
            else:
                days_old = np.full(len(features_df), 999.0)

            # Normalize age: 0.0 = fresh (0 days), 1.0 = 90 days old (quarterly)
            features_df["fundamental_age"] = days_old / 90.0

            for col in fundamental_cols:
                if col in fund_aligned.columns:
                    features_df[col] = fund_aligned[col].fillna(0.0)
                else:
                    features_df[col] = 0.0
        else:
            # No fundamentals - use zeros and max age
            features_df["fundamental_age"] = 1.0  # Max age (90+ days)
            for col in fundamental_cols:
                features_df[col] = 0.0

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

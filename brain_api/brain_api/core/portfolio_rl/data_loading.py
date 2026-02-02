"""Data loading utilities for portfolio RL training.

Provides functions to load historical signals (news, fundamentals) and
align them to weekly frequency for PPO/SAC training.
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from brain_api.core.fundamentals import (
    get_default_data_path,
    load_historical_fundamentals_from_cache,
)

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
        parquet_path: Path to daily_sentiment.parquet

    Returns:
        Dict mapping symbol -> DataFrame with 'sentiment_score' column and DatetimeIndex
    """
    if parquet_path is None:
        parquet_path = get_default_data_path() / "output" / "daily_sentiment.parquet"

    sentiment: dict[str, pd.DataFrame] = {}

    if not parquet_path.exists():
        print(
            f"[PortfolioRL] Warning: News sentiment parquet not found at {parquet_path}"
        )
        return sentiment

    try:
        df = pd.read_parquet(parquet_path)
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
        print(f"[PortfolioRL] Error loading news sentiment: {e}")

    return sentiment


def align_signals_to_weekly(
    prices_dict: dict[str, pd.DataFrame],
    news_sentiment: dict[str, pd.DataFrame],
    fundamentals: dict[str, pd.DataFrame],
    symbols: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    """Align news and fundamentals signals to weekly frequency.

    Args:
        prices_dict: Dict of symbol -> OHLCV DataFrame with DatetimeIndex
        news_sentiment: Dict of symbol -> sentiment DataFrame
        fundamentals: Dict of symbol -> fundamentals DataFrame
        symbols: Ordered list of symbols

    Returns:
        Dict of symbol -> dict of signal_name -> weekly numpy array
    """
    signals: dict[str, dict[str, np.ndarray]] = {}

    for symbol in symbols:
        if symbol not in prices_dict:
            continue

        price_df = prices_dict[symbol]
        if price_df is None or len(price_df) == 0:
            continue

        # Resample to weekly (Friday close)
        weekly_index = price_df["close"].resample("W-FRI").last().dropna().index
        # Normalize to timezone-naive for consistent comparisons
        if weekly_index.tz is not None:
            weekly_index = weekly_index.tz_localize(None)
        n_weeks = len(weekly_index)

        if n_weeks < 2:
            continue

        # Initialize signals with zeros
        symbol_signals: dict[str, np.ndarray] = {
            "news_sentiment": np.zeros(n_weeks),
            "gross_margin": np.zeros(n_weeks),
            "operating_margin": np.zeros(n_weeks),
            "net_margin": np.zeros(n_weeks),
            "current_ratio": np.zeros(n_weeks),
            "debt_to_equity": np.zeros(n_weeks),
            "fundamental_age": np.ones(n_weeks),  # Default to max age
        }

        # Align news sentiment (forward-fill daily to weekly)
        if symbol in news_sentiment:
            sentiment_df = news_sentiment[symbol]
            # Normalize timezone to match weekly_index
            if sentiment_df.index.tz is not None:
                sentiment_df = sentiment_df.copy()
                sentiment_df.index = sentiment_df.index.tz_localize(None)
            # Reindex to weekly and forward-fill
            sentiment_weekly = sentiment_df.reindex(weekly_index, method="ffill")
            symbol_signals["news_sentiment"] = (
                sentiment_weekly["sentiment_score"].fillna(0.0).values
            )

        # Align fundamentals (forward-fill quarterly to weekly)
        if symbol in fundamentals:
            fund_df = fundamentals[symbol]
            # Normalize timezone to match weekly_index
            if fund_df.index.tz is not None:
                fund_df = fund_df.copy()
                fund_df.index = fund_df.index.tz_localize(None)
            fund_aligned = fund_df.reindex(weekly_index, method="ffill")

            for col in [
                "gross_margin",
                "operating_margin",
                "net_margin",
                "current_ratio",
                "debt_to_equity",
            ]:
                if col in fund_aligned.columns:
                    symbol_signals[col] = fund_aligned[col].fillna(0.0).values

            # Compute fundamental_age (days since last update / 90)
            fund_dates = fund_df.index.values
            if len(fund_dates) > 0:
                positions = np.searchsorted(
                    fund_dates, weekly_index.values, side="right"
                )
                valid_positions = np.clip(positions - 1, 0, len(fund_dates) - 1)
                last_updates = fund_dates[valid_positions]
                days_old = (
                    (weekly_index.values - last_updates)
                    .astype("timedelta64[D]")
                    .astype(float)
                )
                days_old[positions == 0] = 999.0
                symbol_signals["fundamental_age"] = days_old / 90.0

        signals[symbol] = symbol_signals

    return signals


def build_rl_training_signals(
    prices_dict: dict[str, pd.DataFrame],
    symbols: list[str],
    start_date: date,
    end_date: date,
) -> dict[str, dict[str, np.ndarray]]:
    """Build complete signals dict for RL training.

    This is the main entry point for loading all historical signals
    and aligning them to weekly frequency.

    Args:
        prices_dict: Dict of symbol -> OHLCV DataFrame
        symbols: List of symbols
        start_date: Training window start
        end_date: Training window end

    Returns:
        Dict of symbol -> dict of signal_name -> weekly numpy array
    """
    print(f"[PortfolioRL] Loading historical signals for {len(symbols)} symbols...")

    # Load news sentiment
    news_sentiment = load_historical_news_sentiment(symbols, start_date, end_date)
    print(f"[PortfolioRL] Loaded news sentiment for {len(news_sentiment)} symbols")

    # Load fundamentals
    fundamentals = load_historical_fundamentals(symbols, start_date, end_date)
    print(f"[PortfolioRL] Loaded fundamentals for {len(fundamentals)} symbols")

    # Align to weekly
    signals = align_signals_to_weekly(
        prices_dict, news_sentiment, fundamentals, symbols
    )
    print(f"[PortfolioRL] Aligned signals for {len(signals)} symbols")

    return signals

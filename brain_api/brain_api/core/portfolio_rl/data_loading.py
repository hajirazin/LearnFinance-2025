"""Data loading utilities for portfolio RL training.

Provides functions to load historical signals (news, fundamentals) and
align them to weekly frequency for SAC training.
"""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from brain_api.core.fundamentals import (
    get_default_data_path,
    load_historical_fundamentals_from_cache,
)
from brain_api.core.news_sentiment import (
    DailyNewsObservation,
    NewsObservationError,
    aggregate_weekly_news_observation,
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
        Dict mapping symbol -> provider-checked daily observation DataFrame.
    """
    if parquet_path is None:
        parquet_path = get_default_data_path() / "output" / "daily_sentiment.parquet"

    sentiment: dict[str, pd.DataFrame] = {}

    if not parquet_path.exists():
        raise FileNotFoundError(f"News sentiment parquet not found at {parquet_path}")

    try:
        df = pd.read_parquet(parquet_path)
        required_columns = {
            "date",
            "symbol",
            "sentiment_score",
            "article_count",
            "avg_confidence",
        }
        missing_columns = required_columns.difference(df.columns)
        if missing_columns:
            raise NewsObservationError(
                f"News parquet missing required columns: {sorted(missing_columns)}"
            )
        df["date"] = pd.to_datetime(df["date"]).dt.date

        for symbol in symbols:
            symbol_df = df[
                (df["symbol"] == symbol)
                & (df["date"] >= start_date)
                & (df["date"] <= end_date)
            ][
                [
                    "date",
                    "sentiment_score",
                    "article_count",
                    "avg_confidence",
                ]
            ].copy()

            if symbol_df.empty:
                raise NewsObservationError(
                    f"No provider-checked news observations for {symbol}"
                )
            symbol_df["date"] = pd.to_datetime(symbol_df["date"])
            symbol_df = symbol_df.set_index("date").sort_index()
            if symbol_df.index.has_duplicates:
                raise NewsObservationError(
                    f"Duplicate daily news observations for {symbol}"
                )
            expected_dates = pd.date_range(start_date, end_date, freq="D")
            missing_dates = expected_dates.difference(symbol_df.index)
            if not missing_dates.empty:
                preview = [value.date().isoformat() for value in missing_dates[:3]]
                raise NewsObservationError(
                    f"Unchecked news gaps for {symbol}: {preview}"
                )
            if symbol_df[["article_count", "avg_confidence"]].isna().any().any():
                raise NewsObservationError(
                    f"Unchecked news rows contain missing coverage for {symbol}"
                )
            sentiment[symbol] = symbol_df
    except (OSError, ValueError, KeyError) as exc:
        raise NewsObservationError(
            f"Failed to load provider-checked news observations: {exc}"
        ) from exc

    return sentiment


def align_signals_to_weekly(
    prices_dict: dict[str, pd.DataFrame],
    news_sentiment: dict[str, pd.DataFrame],
    fundamentals: dict[str, pd.DataFrame],
    symbols: list[str],
    weekly_cutoffs: pd.DatetimeIndex | None = None,
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

        weekly_index = (
            pd.DatetimeIndex(weekly_cutoffs)
            if weekly_cutoffs is not None
            else price_df["close"].resample("W-FRI").last().dropna().index
        )
        # Normalize to timezone-naive for consistent comparisons
        if weekly_index.tz is not None:
            weekly_index = weekly_index.tz_localize(None)
        n_weeks = len(weekly_index)

        if n_weeks < 2:
            continue

        if symbol not in news_sentiment:
            raise NewsObservationError(f"Missing news observations for {symbol}")
        if symbol not in fundamentals:
            raise ValueError(f"Missing point-in-time fundamentals for {symbol}")

        sentiment_df = news_sentiment[symbol]
        if sentiment_df.index.tz is not None:
            sentiment_df = sentiment_df.copy()
            sentiment_df.index = sentiment_df.index.tz_localize(None)
        weekly_news = []
        for weekly_timestamp in weekly_index:
            as_of = weekly_timestamp.date()
            window_start = pd.Timestamp(as_of - timedelta(days=6))
            window = sentiment_df.loc[window_start:weekly_timestamp]
            if len(window) != 7:
                raise NewsObservationError(
                    f"Unchecked 7-day news window for {symbol} ending {as_of}"
                )
            observation = aggregate_weekly_news_observation(
                (
                    DailyNewsObservation(
                        observation_date=index.date(),
                        sentiment_score=float(row["sentiment_score"]),
                        article_count=int(row["article_count"]),
                        avg_confidence=float(row["avg_confidence"]),
                    )
                    for index, row in window.iterrows()
                ),
                as_of_date=as_of,
            )
            weekly_news.append(observation)

        symbol_signals: dict[str, np.ndarray] = {
            "news_sentiment": np.asarray(
                [observation.sentiment_score for observation in weekly_news]
            ),
            "news_coverage": np.asarray(
                [observation.coverage for observation in weekly_news]
            ),
        }

        # Align fundamentals (forward-fill quarterly to weekly)
        fund_df = fundamentals[symbol]
        if fund_df.index.tz is not None:
            fund_df = fund_df.copy()
            fund_df.index = fund_df.index.tz_localize(None)
        fund_aligned = fund_df.reindex(weekly_index, method="ffill")
        ratio_columns = [
            "gross_margin",
            "operating_margin",
            "net_margin",
            "current_ratio",
            "debt_to_equity",
        ]
        missing_ratio_columns = set(ratio_columns).difference(fund_aligned.columns)
        if missing_ratio_columns:
            raise ValueError(
                f"Fundamentals missing columns for {symbol}: "
                f"{sorted(missing_ratio_columns)}"
            )
        if fund_aligned[ratio_columns].isna().any().any():
            raise ValueError(f"Missing point-in-time fundamental ratios for {symbol}")
        for column in ratio_columns:
            symbol_signals[column] = fund_aligned[column].to_numpy(dtype=float)

        fund_dates = fund_df.index.values
        positions = np.searchsorted(fund_dates, weekly_index.values, side="right")
        if np.any(positions == 0):
            raise ValueError(
                f"No filing was available before a training week for {symbol}"
            )
        last_updates = fund_dates[positions - 1]
        symbol_signals["fundamental_age"] = (
            (weekly_index.values - last_updates).astype("timedelta64[D]").astype(float)
        )

        signals[symbol] = symbol_signals

    return signals


def build_rl_training_signals(
    prices_dict: dict[str, pd.DataFrame],
    symbols: list[str],
    start_date: date,
    end_date: date,
    weekly_cutoffs: pd.DatetimeIndex | None = None,
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
    news_start = start_date
    news_end = end_date
    if weekly_cutoffs is not None and len(weekly_cutoffs) > 0:
        news_start = weekly_cutoffs[0].date()
        news_end = weekly_cutoffs[-1].date()
    news_sentiment = load_historical_news_sentiment(
        symbols, news_start - timedelta(days=6), news_end
    )
    print(f"[PortfolioRL] Loaded news sentiment for {len(news_sentiment)} symbols")

    # Load fundamentals
    fundamentals = load_historical_fundamentals(symbols, date.min, news_end)
    print(f"[PortfolioRL] Loaded fundamentals for {len(fundamentals)} symbols")

    # Align to weekly
    signals = align_signals_to_weekly(
        prices_dict,
        news_sentiment,
        fundamentals,
        symbols,
        weekly_cutoffs=weekly_cutoffs,
    )
    print(f"[PortfolioRL] Aligned signals for {len(signals)} symbols")

    return signals

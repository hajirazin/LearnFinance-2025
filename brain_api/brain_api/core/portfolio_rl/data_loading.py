"""Data loading utilities for portfolio RL training.

Provides functions to load historical news and align news plus price momentum
to weekly frequency for SAC training.
"""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from brain_api.core.news_sentiment import (
    DailyNewsObservation,
    NewsObservationError,
    aggregate_weekly_news_observation,
)
from brain_api.core.sac.momentum_signals import (
    MomentumSignalError,
    compute_momentum_1w,
    compute_momentum_4w,
    compute_momentum_12_1,
    compute_realized_vol_20d,
)


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
        parquet_path = (
            Path(__file__).parent.parent.parent.parent
            / "data"
            / "output"
            / "daily_sentiment.parquet"
        )

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
    symbols: list[str],
    weekly_cutoffs: pd.DatetimeIndex | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Align news and price-momentum signals to weekly frequency.

    Args:
        prices_dict: Dict of symbol -> OHLCV DataFrame with DatetimeIndex
        news_sentiment: Dict of symbol -> sentiment DataFrame
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
        }

        # momentum_1w = P_t/P_t-5-1 (5 trading bars); momentum_4w =
        # P_t/P_t-20-1 (20 trading bars); momentum_12_1 = P_t-21/P_t-252-1
        # (skip 21 bars, then 252-bar/~12-month lookback). Same daily
        # close series already loaded for `weekly_index` above -- locate
        # each week's trading-day position in the raw (unresampled)
        # daily series so the bar counts above are literal trading days,
        # not calendar weeks.
        close_series = price_df["close"]
        if close_series.index.tz is not None:
            close_series = close_series.tz_localize(None)
        close_dates = close_series.index.values
        close_values = close_series.to_numpy(dtype=float)
        close_positions = (
            np.searchsorted(close_dates, weekly_index.values, side="right") - 1
        )
        if np.any(close_positions < 0):
            raise ValueError(
                f"No daily close available on/before a training week for {symbol}"
            )

        momentum_1w = np.empty(n_weeks)
        momentum_4w = np.empty(n_weeks)
        momentum_12_1 = np.empty(n_weeks)
        realized_vol_20d = np.empty(n_weeks)
        for week_idx, close_position in enumerate(close_positions):
            as_of_index = int(close_position)
            try:
                momentum_1w[week_idx] = compute_momentum_1w(
                    close_values, as_of_index=as_of_index
                )
                momentum_4w[week_idx] = compute_momentum_4w(
                    close_values, as_of_index=as_of_index
                )
                momentum_12_1[week_idx] = compute_momentum_12_1(
                    close_values, as_of_index=as_of_index
                )
                realized_vol_20d[week_idx] = compute_realized_vol_20d(
                    close_values, as_of_index=as_of_index
                )
            except MomentumSignalError:
                # Match live eligibility: insufficient history makes the
                # symbol ineligible for that week (NaN → mask), never a
                # silent zero-fill of the feature values.
                momentum_1w[week_idx] = np.nan
                momentum_4w[week_idx] = np.nan
                momentum_12_1[week_idx] = np.nan
                realized_vol_20d[week_idx] = np.nan

        symbol_signals["momentum_1w"] = momentum_1w
        symbol_signals["momentum_4w"] = momentum_4w
        symbol_signals["momentum_12_1"] = momentum_12_1
        symbol_signals["realized_vol_20d"] = realized_vol_20d

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

    # Align to weekly
    signals = align_signals_to_weekly(
        prices_dict,
        news_sentiment,
        symbols,
        weekly_cutoffs=weekly_cutoffs,
    )
    print(f"[PortfolioRL] Aligned signals for {len(signals)} symbols")

    return signals

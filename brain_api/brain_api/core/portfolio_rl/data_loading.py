"""Data loading utilities for portfolio RL training.

Provides functions to load historical signals (news, fundamentals) and
align them to weekly frequency for PPO/SAC training.
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


def get_default_data_path() -> Path:
    """Get the default data path for brain_api."""
    return Path(__file__).parent.parent.parent.parent / "data"


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


def load_historical_fundamentals(
    symbols: list[str],
    start_date: date,
    end_date: date,
    cache_path: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Load historical fundamentals from Alpha Vantage cache.

    Args:
        symbols: List of ticker symbols
        start_date: Start of data window
        end_date: End of data window
        cache_path: Base path for fundamentals cache

    Returns:
        Dict mapping symbol -> DataFrame with fundamental ratio columns
    """
    if cache_path is None:
        cache_path = get_default_data_path()

    fundamentals: dict[str, pd.DataFrame] = {}

    try:
        from brain_api.core.fundamentals import (
            compute_ratios,
            load_raw_response,
            parse_quarterly_statements,
        )
    except ImportError:
        print("[PortfolioRL] Warning: Could not import fundamentals utilities")
        return fundamentals

    for symbol in symbols:
        try:
            income_data = load_raw_response(cache_path, symbol, "income_statement")
            balance_data = load_raw_response(cache_path, symbol, "balance_sheet")

            if income_data is None and balance_data is None:
                continue

            income_stmts = []
            balance_stmts = []

            if income_data:
                income_stmts = parse_quarterly_statements(
                    symbol, "income_statement", income_data
                )
            if balance_data:
                balance_stmts = parse_quarterly_statements(
                    symbol, "balance_sheet", balance_data
                )

            rows = []
            fiscal_dates = set()

            for stmt in income_stmts:
                if (
                    start_date
                    <= date.fromisoformat(stmt.fiscal_date_ending)
                    <= end_date
                ):
                    fiscal_dates.add(stmt.fiscal_date_ending)
            for stmt in balance_stmts:
                if (
                    start_date
                    <= date.fromisoformat(stmt.fiscal_date_ending)
                    <= end_date
                ):
                    fiscal_dates.add(stmt.fiscal_date_ending)

            for fiscal_date in sorted(fiscal_dates):
                income_stmt = next(
                    (s for s in income_stmts if s.fiscal_date_ending == fiscal_date),
                    None,
                )
                balance_stmt = next(
                    (s for s in balance_stmts if s.fiscal_date_ending == fiscal_date),
                    None,
                )

                ratios = compute_ratios(income_stmt, balance_stmt)
                if ratios:
                    rows.append(
                        {
                            "date": pd.to_datetime(fiscal_date),
                            "gross_margin": ratios.gross_margin,
                            "operating_margin": ratios.operating_margin,
                            "net_margin": ratios.net_margin,
                            "current_ratio": ratios.current_ratio,
                            "debt_to_equity": ratios.debt_to_equity,
                        }
                    )

            if rows:
                df = pd.DataFrame(rows).set_index("date").sort_index()
                fundamentals[symbol] = df

        except Exception as e:
            print(f"[PortfolioRL] Error loading fundamentals for {symbol}: {e}")
            continue

    return fundamentals


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
            # Reindex to weekly and forward-fill
            sentiment_weekly = sentiment_df.reindex(weekly_index, method="ffill")
            symbol_signals["news_sentiment"] = (
                sentiment_weekly["sentiment_score"].fillna(0.0).values
            )

        # Align fundamentals (forward-fill quarterly to weekly)
        if symbol in fundamentals:
            fund_df = fundamentals[symbol]
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

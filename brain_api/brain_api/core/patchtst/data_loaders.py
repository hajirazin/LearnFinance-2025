"""Data loading utilities for PatchTST training and inference."""

from datetime import date
from pathlib import Path

import pandas as pd

from brain_api.core.features import compute_ohlcv_log_returns
from brain_api.core.patchtst.config import PatchTSTConfig


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
        parquet_path = Path(__file__).parent.parent.parent.parent / "data" / "output" / "daily_sentiment.parquet"

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
                (df["symbol"] == symbol) &
                (df["date"] >= start_date) &
                (df["date"] <= end_date)
            ][["date", "sentiment_score"]].copy()

            if len(symbol_df) > 0:
                symbol_df["date"] = pd.to_datetime(symbol_df["date"])
                symbol_df = symbol_df.set_index("date").sort_index()
                sentiment[symbol] = symbol_df

    except Exception as e:
        print(f"[PatchTST] Error loading news sentiment: {e}")

    return sentiment


def load_historical_fundamentals(
    symbols: list[str],
    start_date: date,
    end_date: date,
    cache_path: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Load historical fundamentals from cache.

    Fundamentals are quarterly data that should be forward-filled to daily.
    Expects cached JSON files from Alpha Vantage at:
    brain_api/data/fundamentals_cache/{symbol}/

    Args:
        symbols: List of ticker symbols
        start_date: Start of data window
        end_date: End of data window
        cache_path: Base path for fundamentals cache

    Returns:
        Dict mapping symbol -> DataFrame with fundamental ratio columns
        and DatetimeIndex (quarterly dates, to be forward-filled later)
    """
    if cache_path is None:
        cache_path = Path(__file__).parent.parent.parent / "data" / "raw" / "fundamentals"

    fundamentals: dict[str, pd.DataFrame] = {}

    # Import fundamentals parsing utilities
    try:
        from brain_api.core.fundamentals import (
            compute_ratios,
            load_raw_response,
            parse_quarterly_statements,
        )
    except ImportError:
        print("[PatchTST] Warning: Could not import fundamentals utilities")
        return fundamentals

    for symbol in symbols:
        try:
            # Load cached responses
            income_data = load_raw_response(cache_path, symbol, "income_statement")
            balance_data = load_raw_response(cache_path, symbol, "balance_sheet")

            if income_data is None and balance_data is None:
                continue

            # Parse statements
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

            # Collect ratios for each fiscal date
            rows = []
            fiscal_dates = set()

            for stmt in income_stmts:
                if start_date <= date.fromisoformat(stmt.fiscal_date_ending) <= end_date:
                    fiscal_dates.add(stmt.fiscal_date_ending)
            for stmt in balance_stmts:
                if start_date <= date.fromisoformat(stmt.fiscal_date_ending) <= end_date:
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
                    rows.append({
                        "date": pd.to_datetime(fiscal_date),
                        "gross_margin": ratios.gross_margin,
                        "operating_margin": ratios.operating_margin,
                        "net_margin": ratios.net_margin,
                        "current_ratio": ratios.current_ratio,
                        "debt_to_equity": ratios.debt_to_equity,
                    })

            if rows:
                df = pd.DataFrame(rows).set_index("date").sort_index()
                fundamentals[symbol] = df

        except Exception as e:
            print(f"[PatchTST] Error loading fundamentals for {symbol}: {e}")
            continue

    return fundamentals


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
        features_df = compute_ohlcv_log_returns(price_df, use_returns=config.use_returns)

        # Add news sentiment (forward-fill missing days)
        if symbol in news_sentiment:
            sentiment_df = news_sentiment[symbol]
            # Reindex to match price dates and forward-fill
            sentiment_aligned = sentiment_df.reindex(features_df.index, method="ffill")
            features_df["news_sentiment"] = sentiment_aligned["sentiment_score"].fillna(0.0)
        else:
            features_df["news_sentiment"] = 0.0  # Neutral if no news data

        # Add fundamentals (forward-fill quarterly data)
        fundamental_cols = ["gross_margin", "operating_margin", "net_margin",
                          "current_ratio", "debt_to_equity"]
        if symbol in fundamentals:
            fund_df = fundamentals[symbol]
            # Reindex to match price dates and forward-fill
            fund_aligned = fund_df.reindex(features_df.index, method="ffill")
            for col in fundamental_cols:
                if col in fund_aligned.columns:
                    features_df[col] = fund_aligned[col].fillna(0.0)
                else:
                    features_df[col] = 0.0
        else:
            # No fundamentals - use zeros
            for col in fundamental_cols:
                features_df[col] = 0.0

        # Ensure column order matches config.feature_names
        features_df = features_df[config.feature_names]

        if len(features_df) >= config.context_length:
            aligned[symbol] = features_df

    return aligned


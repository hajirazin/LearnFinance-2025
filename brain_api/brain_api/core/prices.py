"""Shared price data loading utilities.

This module provides common price fetching functionality used by
both LSTM and PatchTST models.
"""

import logging
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def load_prices_yfinance(
    symbols: list[str],
    start_date: date,
    end_date: date,
    log_prefix: str = "[Prices]",
) -> dict[str, pd.DataFrame]:
    """Load OHLCV price data for symbols using yfinance.

    Attempts batch download first for efficiency, then falls back to
    individual symbol fetching if batch fails.

    Args:
        symbols: List of ticker symbols
        start_date: Start of data window
        end_date: End of data window
        log_prefix: Prefix for log messages (e.g., "[LSTM]" or "[PatchTST]")

    Returns:
        Dict mapping symbol -> DataFrame with OHLCV columns and DatetimeIndex
    """
    prices: dict[str, pd.DataFrame] = {}
    failed_symbols: list[str] = []

    print(
        f"{log_prefix} Downloading prices for {len(symbols)} symbols from yfinance..."
    )
    print(f"{log_prefix} Date range: {start_date} to {end_date}")

    # Try batch download first
    try:
        tickers_str = " ".join(symbols)
        data = yf.download(
            tickers_str,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            progress=False,
            group_by="ticker",
        )

        # Check if download returned valid data
        if data is not None and not data.empty and hasattr(data, "columns"):
            if len(symbols) == 1:
                symbol = symbols[0]
                try:
                    df = data[["Open", "High", "Low", "Close", "Volume"]].copy()
                    df.columns = ["open", "high", "low", "close", "volume"]
                    df = df.dropna()
                    if len(df) > 0:
                        prices[symbol] = df
                    else:
                        failed_symbols.append(symbol)
                except (KeyError, TypeError) as e:
                    print(f"{log_prefix} Failed to parse {symbol}: {e}")
                    failed_symbols.append(symbol)
            else:
                # Multiple tickers: data is multi-level columns
                try:
                    available_tickers = set(data.columns.get_level_values(0))
                    for symbol in symbols:
                        if symbol in available_tickers:
                            try:
                                df = data[symbol][
                                    ["Open", "High", "Low", "Close", "Volume"]
                                ].copy()
                                df.columns = ["open", "high", "low", "close", "volume"]
                                df = df.dropna()
                                if len(df) > 0:
                                    prices[symbol] = df
                                else:
                                    failed_symbols.append(symbol)
                            except (KeyError, TypeError) as e:
                                print(f"{log_prefix} Failed to parse {symbol}: {e}")
                                failed_symbols.append(symbol)
                        else:
                            failed_symbols.append(symbol)
                except (AttributeError, TypeError) as e:
                    print(f"{log_prefix} Batch download parsing failed: {e}")
                    failed_symbols = list(symbols)
        else:
            print(f"{log_prefix} Batch download returned empty or invalid data")
            failed_symbols = list(symbols)

    except Exception as e:
        print(f"{log_prefix} Batch download failed: {e}")
        failed_symbols = list(symbols)

    # Fallback: fetch failed symbols individually
    if failed_symbols:
        print(f"{log_prefix} Fetching {len(failed_symbols)} symbols individually...")
        for symbol in failed_symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=start_date.isoformat(),
                    end=end_date.isoformat(),
                )
                if df is not None and not df.empty:
                    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                    df.columns = ["open", "high", "low", "close", "volume"]
                    df = df.dropna()
                    if len(df) > 0:
                        prices[symbol] = df
                        print(f"{log_prefix} ✓ {symbol}: {len(df)} days")
                    else:
                        print(f"{log_prefix} ✗ {symbol}: no data after dropna")
                else:
                    print(f"{log_prefix} ✗ {symbol}: empty response")
            except Exception as e:
                print(f"{log_prefix} ✗ {symbol}: {type(e).__name__}: {e}")

    # Summary
    successful = len(prices)
    failed = len(symbols) - successful
    print(
        f"{log_prefix} Price download complete: {successful}/{len(symbols)} symbols loaded"
    )
    if failed > 0:
        missing = [s for s in symbols if s not in prices]
        print(f"{log_prefix} Missing symbols: {missing}")

    return prices


def compute_min_walkforward_days(cutoff_date: date) -> int:
    """Minimum trading days a symbol needs for walk-forward feasibility.

    Derived from the training window: the symbol must have price data
    going back to before training_start minus an LSTM lookback buffer
    (61 trading days ≈ 200 calendar days with margin).

    Args:
        cutoff_date: The reference/end date (typically the current cutoff).

    Returns:
        Minimum number of trading days required.
    """
    from brain_api.core.config import resolve_training_window

    start_date, _ = resolve_training_window()
    calendar_span = (cutoff_date - start_date).days + 200
    return int(calendar_span * 252 / 365)


def filter_symbols_by_min_history(
    symbols: list[str],
    min_trading_days: int,
    reference_date: date,
) -> tuple[list[str], list[tuple[str, int]]]:
    """Filter symbols to those with sufficient price history.

    Downloads close prices going back far enough to cover min_trading_days
    of trading data, then checks each symbol's actual row count.

    Args:
        symbols: Candidate ticker symbols.
        min_trading_days: Minimum number of trading days required.
        reference_date: End date for the history check (typically cutoff_date).

    Returns:
        Tuple of (qualifying_symbols, excluded_with_day_counts) where
        excluded_with_day_counts is a list of (symbol, actual_days) tuples.
    """
    if not symbols:
        return [], []

    # ~2x calendar days covers weekends/holidays with margin
    start_date = reference_date - timedelta(days=min_trading_days * 2)

    logger.info(
        f"[HistoryFilter] Checking price history for {len(symbols)} symbols "
        f"(min {min_trading_days} trading days, window {start_date} to {reference_date})"
    )

    prices = load_prices_yfinance(
        symbols, start_date, reference_date, log_prefix="[HistoryFilter]"
    )

    qualifying: list[str] = []
    excluded: list[tuple[str, int]] = []

    for symbol in symbols:
        df = prices.get(symbol)
        actual_days = len(df) if df is not None else 0
        if actual_days >= min_trading_days:
            qualifying.append(symbol)
        else:
            excluded.append((symbol, actual_days))

    logger.info(
        f"[HistoryFilter] {len(qualifying)} symbols qualify, "
        f"{len(excluded)} excluded for insufficient history"
    )

    return qualifying, excluded

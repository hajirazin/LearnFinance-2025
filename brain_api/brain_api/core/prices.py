"""Shared price data loading utilities.

This module provides common price fetching functionality used by
both LSTM and PatchTST models.
"""

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_YFINANCE_IO_LOCK = threading.Lock()
_OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def repair_ohlc_envelope(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy whose high/low enclose each finite OHLC candle body.

    Invalid values are deliberately not imputed: NumPy propagates NaN and
    callers retain their existing reject/drop semantics for non-finite or
    non-positive evidence.
    """
    required = ("open", "high", "low", "close")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"OHLC frame missing columns {missing}")
    repaired = frame.copy()
    open_ = repaired["open"].to_numpy(dtype=np.float64)
    high = repaired["high"].to_numpy(dtype=np.float64)
    low = repaired["low"].to_numpy(dtype=np.float64)
    close = repaired["close"].to_numpy(dtype=np.float64)
    valid = (
        np.isfinite(open_)
        & np.isfinite(high)
        & np.isfinite(low)
        & np.isfinite(close)
        & (open_ > 0)
        & (high > 0)
        & (low > 0)
        & (close > 0)
    )
    repaired_low = low.copy()
    repaired_high = high.copy()
    repaired_low[valid] = np.minimum(low[valid], np.minimum(open_[valid], close[valid]))
    repaired_high[valid] = np.maximum(
        high[valid], np.maximum(open_[valid], close[valid])
    )
    repaired["low"] = repaired_low
    repaired["high"] = repaired_high
    return repaired


def _symbol_ohlcv_from_yahoo_download(data: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Return one symbol's lowercase OHLCV from a yfinance download DataFrame."""
    frame = data[symbol] if isinstance(data.columns, pd.MultiIndex) else data
    ohlcv = frame.loc[:, _OHLCV_COLUMNS].copy()
    ohlcv.columns = ["open", "high", "low", "close", "volume"]
    return repair_ohlc_envelope(ohlcv).dropna()


@contextmanager
def yfinance_io_lock() -> Iterator[None]:
    """Serialize all yfinance I/O in this process.

    yfinance keeps process-global download state. Concurrent FastAPI
    threads (PatchTST + SAC prices + news) must not interleave Yahoo
    calls or histories get truncated to whichever download finished last.
    """
    with _YFINANCE_IO_LOCK:
        yield


def load_prices_yfinance(
    symbols: list[str],
    start_date: date,
    end_date: date,
    log_prefix: str = "[Prices]",
) -> dict[str, pd.DataFrame]:
    """Load OHLCV price data for symbols using yfinance.

    Attempts batch download first for efficiency, then falls back to
    individual symbol fetching if batch fails. Holds the process
    yfinance I/O lock for the entire download so concurrent callers
    cannot corrupt each other's histories.
    """
    with yfinance_io_lock():
        return _load_prices_yfinance_unlocked(
            symbols, start_date, end_date, log_prefix=log_prefix
        )


def _load_prices_yfinance_unlocked(
    symbols: list[str],
    start_date: date,
    end_date: date,
    log_prefix: str = "[Prices]",
) -> dict[str, pd.DataFrame]:
    prices: dict[str, pd.DataFrame] = {}
    failed_symbols: list[str] = []
    yahoo_end_exclusive = end_date + timedelta(days=1)

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
            end=yahoo_end_exclusive.isoformat(),
            progress=False,
            group_by="ticker",
            auto_adjust=True,
            threads=False,
        )

        # Check if download returned valid data
        if data is not None and not data.empty and hasattr(data, "columns"):
            for symbol in symbols:
                try:
                    df = _symbol_ohlcv_from_yahoo_download(data, symbol)
                    if len(df) > 0:
                        prices[symbol] = df
                    else:
                        failed_symbols.append(symbol)
                except (KeyError, TypeError, IndexError) as e:
                    print(f"{log_prefix} Failed to parse {symbol}: {e}")
                    failed_symbols.append(symbol)
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
                    end=yahoo_end_exclusive.isoformat(),
                    auto_adjust=True,
                )
                if df is not None and not df.empty:
                    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                    df.columns = ["open", "high", "low", "close", "volume"]
                    df = repair_ohlc_envelope(df).dropna()
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

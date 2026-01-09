"""Shared price data loading utilities.

This module provides common price fetching functionality used by
both LSTM and PatchTST models.
"""

from datetime import date

import pandas as pd
import yfinance as yf


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

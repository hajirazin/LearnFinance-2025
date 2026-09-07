"""Cutoff filters shared by LSTM and PatchTST snapshot backfills."""

from datetime import date

import pandas as pd


def _filter_prices_by_cutoff(
    prices: dict[str, pd.DataFrame],
    cutoff_date: date,
) -> dict[str, pd.DataFrame]:
    """Filter price frames to rows on or before ``cutoff_date``.

    The cutoff is localized to a frame's timezone before comparison so
    yfinance's timezone-aware indexes remain compatible with a naive date.
    Symbols with no remaining rows are omitted.
    """
    cutoff_ts = pd.Timestamp(cutoff_date)
    out: dict[str, pd.DataFrame] = {}
    for symbol, df in prices.items():
        symbol_cutoff = cutoff_ts
        if df.index.tz is not None and symbol_cutoff.tz is None:
            symbol_cutoff = symbol_cutoff.tz_localize(df.index.tz)
        filtered = df[df.index <= symbol_cutoff]
        if len(filtered) > 0:
            out[symbol] = filtered.copy()
    return out


def _filter_signals_by_cutoff(
    signals: dict[str, pd.DataFrame],
    cutoff_date: date,
) -> dict[str, pd.DataFrame]:
    """Filter signal frames to rows on or before ``cutoff_date``.

    Datetime indexes retain timezone-aware comparison semantics. Unparseable
    non-datetime indexes preserve their original frames. Empty results are
    omitted.
    """
    cutoff_ts = pd.Timestamp(cutoff_date)
    result: dict[str, pd.DataFrame] = {}

    for symbol, df in signals.items():
        if df.empty:
            continue

        if isinstance(df.index, pd.DatetimeIndex):
            symbol_cutoff = cutoff_ts
            if df.index.tz is not None and symbol_cutoff.tz is None:
                symbol_cutoff = symbol_cutoff.tz_localize(df.index.tz)
            filtered = df[df.index <= symbol_cutoff]
        else:
            try:
                idx = pd.to_datetime(df.index)
                symbol_cutoff = cutoff_ts
                if (
                    isinstance(idx, pd.DatetimeIndex)
                    and idx.tz is not None
                    and symbol_cutoff.tz is None
                ):
                    symbol_cutoff = symbol_cutoff.tz_localize(idx.tz)
                filtered = df[idx <= symbol_cutoff]
            except (ValueError, TypeError):
                filtered = df

        if len(filtered) > 0:
            result[symbol] = filtered.copy()

    return result

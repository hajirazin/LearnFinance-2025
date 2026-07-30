"""Daily market-data loading for SAC walk-forward forecaster inference."""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

logger = logging.getLogger(__name__)


def load_daily_ohlcv(
    symbol: str, start_date: date, end_date: date
) -> pd.DataFrame | None:
    """Load a bounded daily OHLCV window for one symbol."""
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start_date.isoformat(),
            end=(end_date + timedelta(days=1)).isoformat(),
            interval="1d",
        )
        if df.empty:
            return None
        df.columns = df.columns.str.lower()
        return df[["open", "high", "low", "close", "volume"]]
    except Exception as exc:
        logger.debug("[WalkForward] Failed to load OHLCV for %s: %s", symbol, exc)
        return None

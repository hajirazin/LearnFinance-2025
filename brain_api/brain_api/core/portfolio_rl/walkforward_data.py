"""Daily market-data loading for SAC walk-forward forecaster inference."""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

from brain_api.core.prices import load_prices_yfinance

logger = logging.getLogger(__name__)


def load_daily_ohlcv(
    symbol: str, start_date: date, end_date: date
) -> pd.DataFrame | None:
    """Load a bounded daily OHLCV window for one symbol."""
    try:
        prices = load_prices_yfinance(
            [symbol],
            start_date,
            end_date + timedelta(days=1),
            log_prefix="[WalkForward]",
        )
        frame = prices.get(symbol)
        if frame is None or frame.empty:
            return None
        return frame[["open", "high", "low", "close", "volume"]]
    except Exception as exc:
        logger.debug("[WalkForward] Failed to load OHLCV for %s: %s", symbol, exc)
        return None

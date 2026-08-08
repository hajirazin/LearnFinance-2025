"""Filter symbols by maximum price using yfinance."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

from brain_api.core.prices import load_prices_yfinance

logger = logging.getLogger(__name__)

MAX_PRICE_INR = 5000.0  # Maximum price threshold in INR


@dataclass(frozen=True)
class MaxPriceExclusion:
    """One symbol excluded by the max-price filter."""

    symbol: str
    price: float | None
    reason: Literal["above_max", "missing_price"]


def filter_symbols_by_max_price(
    symbols: list[str],
    *,
    as_of: date | None = None,
    max_price: float = MAX_PRICE_INR,
) -> tuple[list[str], list[MaxPriceExclusion]]:
    """Filter symbols to those with price at/below max_price as of a date.

    Args:
        symbols: Candidate ticker symbols.
        as_of: Reference date for the price lookback window. Defaults to today.
            Score-batch / universe builders should pass the inference cutoff.
        max_price: Maximum allowed close price (INR for India).

    Returns:
        Tuple of (qualifying_symbols, exclusions). Missing downloads are
        ``reason="missing_price"`` (price=None); over-max are
        ``reason="above_max"``. No silent 0.0 fake prices.
    """
    if not symbols:
        return [], []

    ref = as_of if as_of is not None else date.today()
    start = ref - timedelta(days=7)

    logger.info(
        f"[PriceFilter] Checking prices for {len(symbols)} symbols "
        f"(max price {max_price} INR, as_of={ref})..."
    )

    prices = load_prices_yfinance(
        symbols,
        start,
        ref,
        log_prefix="[PriceFilter]",
    )

    qualifying: list[str] = []
    excluded: list[MaxPriceExclusion] = []

    for symbol in symbols:
        df = prices.get(symbol)
        if df is None or df.empty:
            excluded.append(
                MaxPriceExclusion(symbol=symbol, price=None, reason="missing_price")
            )
            continue

        actual_price = float(df["close"].iloc[-1])
        if actual_price <= max_price:
            qualifying.append(symbol)
        else:
            excluded.append(
                MaxPriceExclusion(symbol=symbol, price=actual_price, reason="above_max")
            )

    logger.info(
        f"[PriceFilter] {len(qualifying)} symbols qualify, "
        f"{len(excluded)} excluded "
        f"(above_max={sum(1 for e in excluded if e.reason == 'above_max')}, "
        f"missing={sum(1 for e in excluded if e.reason == 'missing_price')})"
    )

    return qualifying, excluded

"""Filter symbols by maximum price using yfinance."""

import logging
from datetime import date, timedelta

from brain_api.core.prices import load_prices_yfinance

logger = logging.getLogger(__name__)

MAX_PRICE_INR = 5000.0  # Example maximum price threshold in INR


def filter_symbols_by_max_price(
    symbols: list[str],
) -> tuple[list[str], list[tuple[str, float]]]:
    """Filter symbols to those with price below max_price.

    Args:
        symbols: Candidate ticker symbols.

    Returns:
        Tuple of (qualifying_symbols, excluded_with_prices) where
        excluded_with_prices is a list of (symbol, actual_price) tuples.
    """
    if not symbols:
        return [], []

    logger.info(
        f"[PriceFilter] Checking prices for {len(symbols)} symbols "
        f"(max price {MAX_PRICE_INR} INR)..."
    )

    prices = load_prices_yfinance(
        symbols,
        date.today() - timedelta(days=7),
        date.today(),
        log_prefix="[PriceFilter]",
    )

    qualifying: list[str] = []
    excluded: list[tuple[str, float]] = []

    for symbol in symbols:
        df = prices.get(symbol)
        actual_price = df["close"].iloc[-1] if df is not None and not df.empty else None
        if actual_price is not None and actual_price <= MAX_PRICE_INR:
            qualifying.append(symbol)
        else:
            excluded.append((symbol, actual_price if actual_price is not None else 0.0))

    logger.info(
        f"[PriceFilter] {len(qualifying)} symbols qualify, "
        f"{len(excluded)} excluded for exceeding max price"
    )

    return qualifying, excluded

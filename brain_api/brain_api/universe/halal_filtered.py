"""Halal_Filtered stock universe: top 15 factor-scored halal stocks.

Pipeline:
    1. Get halal_new base universe (~410 stocks from 5 ETFs + Alpaca filter)
    2. Fetch yfinance metrics (fundamentals + 6-month price history)
    3. Apply junk filter (ROE > 0, Price > SMA200, Beta < 2)
    4. Compute factor scores (0.4*Momentum + 0.3*Quality + 0.3*Value)
    5. Return top 15 by factor score

This produces the same count (15) as the halal top-15 universe,
making it safe for RL allocators that require exactly 15 stocks.
"""

import logging
from datetime import UTC, datetime

from brain_api.universe.cache import load_cached_universe, save_universe_cache
from brain_api.universe.halal_new import get_halal_new_universe
from brain_api.universe.stock_filter import (
    apply_junk_filter,
    compute_factor_scores,
    fetch_stock_metrics,
)

logger = logging.getLogger(__name__)

HALAL_FILTERED_TOP_N = 15


def get_halal_filtered_universe() -> dict:
    """Build filtered + scored halal universe (top 15).

    Returns:
        Dict with:
        - stocks: top 15 factor-scored holdings (with metrics, score, components)
        - total_before_filter: count before junk filter
        - total_after_filter: count after junk filter
        - total_scored: count after scoring (same as after filter)
        - top_n: how many stocks were selected
        - fetched_at: ISO timestamp
    """
    cached = load_cached_universe("halal_filtered")
    if cached is not None:
        return cached

    halal_new = get_halal_new_universe()
    stocks = halal_new["stocks"]
    symbols = [s["symbol"] for s in stocks]

    logger.info(f"Halal_Filtered: starting with {len(symbols)} stocks from halal_new")

    metrics = fetch_stock_metrics(symbols)

    passed, failed = apply_junk_filter(stocks, metrics)
    logger.info(
        f"Halal_Filtered: junk filter passed={len(passed)}, failed={len(failed)}"
    )

    scored = compute_factor_scores(passed)

    top_n = scored[:HALAL_FILTERED_TOP_N]
    logger.info(
        f"Halal_Filtered: returning top {len(top_n)} of {len(scored)} scored stocks"
    )

    result = {
        "stocks": top_n,
        "total_before_filter": len(stocks),
        "total_after_filter": len(passed),
        "total_scored": len(scored),
        "top_n": HALAL_FILTERED_TOP_N,
        "fetched_at": datetime.now(UTC).isoformat(),
    }
    save_universe_cache("halal_filtered", result)
    return result


def get_halal_filtered_symbols() -> list[str]:
    """Get just the list of Halal_Filtered stock symbols (top 15).

    Convenience function for use by training pipelines.

    Returns:
        List of top 15 factor-scored halal stock symbols.
    """
    universe = get_halal_filtered_universe()
    return [s["symbol"] for s in universe["stocks"]]

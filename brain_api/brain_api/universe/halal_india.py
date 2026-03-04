"""Halal_India stock universe: top 15 by India PatchTST predicted weekly return.

Pipeline:
    1. Get NiftyShariah500 base universe (~210 stocks, already .NS-suffixed)
    2. Pre-filter: exclude symbols without enough price history for
       walk-forward training (threshold derived from training window config)
    3. Run PatchTST batch inference with India storage (OHLCV only)
    4. Filter out predictions with no valid return
    5. Return top 15 by predicted_weekly_return_pct descending

Symbols carry .NS suffix throughout the entire pipeline -- no append/strip
transformations are needed because NiftyShariah500 provides yfinance-ready
symbols from the start.
"""

import logging
import threading
from datetime import UTC, datetime

from brain_api.core.config import resolve_cutoff_date
from brain_api.core.patchtst.inference import run_batch_inference
from brain_api.core.prices import (
    compute_min_walkforward_days,
    filter_symbols_by_min_history,
)
from brain_api.storage.patchtst.local import PatchTSTIndiaModelStorage
from brain_api.universe.cache import load_cached_universe, save_universe_cache
from brain_api.universe.nifty_shariah_500 import get_nifty_shariah_500_universe

logger = logging.getLogger(__name__)

HALAL_INDIA_TOP_N = 15
NS_SUFFIX = ".NS"


def get_halal_india_universe(
    shutdown_event: threading.Event | None = None,
) -> dict:
    """Build PatchTST forecast-based India halal universe (top 15).

    Symbols without enough price history for walk-forward training are
    excluded before PatchTST inference. The threshold is derived
    dynamically from the training window config.

    Args:
        shutdown_event: Reserved for future cancellation support.

    Returns:
        Dict with:
        - stocks: top 15 by predicted weekly return (with rank, .NS-suffixed)
        - total_candidates: symbols with valid predictions
        - total_universe: total NiftyShariah500 symbols
        - filtered_insufficient_history: count excluded for short history
        - top_n: how many stocks were selected
        - selection_method: "patchtst_forecast"
        - model_version: India PatchTST version used
        - symbol_suffix: ".NS" (informational -- symbols already include it)
        - fetched_at: ISO timestamp

    Raises:
        ValueError: If no promoted India PatchTST model is available.
    """
    cached = load_cached_universe("halal_india")
    if cached is not None:
        return cached

    base = get_nifty_shariah_500_universe(shutdown_event=shutdown_event)
    symbols = [s["symbol"] for s in base["stocks"]]
    total_universe = len(symbols)

    cutoff_date = resolve_cutoff_date()
    min_trading_days = compute_min_walkforward_days(cutoff_date)

    qualifying_symbols, excluded = filter_symbols_by_min_history(
        symbols, min_trading_days, cutoff_date
    )

    if excluded:
        for sym, days in excluded:
            logger.warning(
                f"Halal_India: excluded {sym} — only {days} trading days "
                f"(need {min_trading_days})"
            )

    logger.info(
        f"Halal_India: {len(qualifying_symbols)}/{total_universe} symbols pass "
        f"min-history filter ({len(excluded)} excluded), "
        f"running India PatchTST inference"
    )

    storage = PatchTSTIndiaModelStorage()
    batch_result = run_batch_inference(qualifying_symbols, cutoff_date, storage=storage)

    valid = [
        p for p in batch_result.predictions if p.predicted_weekly_return_pct is not None
    ]
    top_n = valid[:HALAL_INDIA_TOP_N]

    logger.info(
        f"Halal_India: {len(valid)} valid predictions, "
        f"returning top {len(top_n)} (model {batch_result.model_version})"
    )

    stocks = [
        {
            "symbol": p.symbol,
            "predicted_weekly_return_pct": p.predicted_weekly_return_pct,
            "rank": rank + 1,
            "model_version": batch_result.model_version,
        }
        for rank, p in enumerate(top_n)
    ]

    result = {
        "stocks": stocks,
        "total_candidates": len(valid),
        "total_universe": total_universe,
        "filtered_insufficient_history": len(excluded),
        "top_n": HALAL_INDIA_TOP_N,
        "selection_method": "patchtst_forecast",
        "model_version": batch_result.model_version,
        "symbol_suffix": NS_SUFFIX,
        "fetched_at": datetime.now(UTC).isoformat(),
    }
    save_universe_cache("halal_india", result)
    return result


def get_halal_india_symbols(
    shutdown_event: threading.Event | None = None,
) -> list[str]:
    """Get just the list of Halal_India stock symbols (top 15, .NS-suffixed).

    Args:
        shutdown_event: Reserved for future cancellation support.

    Returns:
        List of top 15 India stock symbols by PatchTST predicted return
        (with .NS suffix, e.g., 'RELIANCE.NS').
    """
    universe = get_halal_india_universe(shutdown_event=shutdown_event)
    return [s["symbol"] for s in universe["stocks"]]

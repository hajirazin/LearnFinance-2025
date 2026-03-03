"""Halal_Filtered stock universe: top 15 by PatchTST predicted weekly return.

Pipeline:
    1. Get halal_new base universe (~410 stocks from 5 ETFs + Alpaca filter)
    2. Pre-filter: exclude symbols without enough price history for
       walk-forward training (threshold derived from training window config)
    3. Run PatchTST batch inference on qualifying symbols (OHLCV only)
    4. Filter out predictions with no valid return
    5. Return top 15 by predicted_weekly_return_pct descending

This produces the same count (15) as the halal top-15 universe,
making it safe for RL allocators that require exactly 15 stocks.
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
from brain_api.universe.cache import load_cached_universe, save_universe_cache
from brain_api.universe.halal_new import get_halal_new_universe

logger = logging.getLogger(__name__)

HALAL_FILTERED_TOP_N = 15


def get_halal_filtered_universe(
    shutdown_event: threading.Event | None = None,
) -> dict:
    """Build PatchTST forecast-based halal universe (top 15).

    Symbols without enough price history for walk-forward training are
    excluded before PatchTST inference. The threshold is derived
    dynamically from the training window config (training span + LSTM
    lookback buffer).

    Args:
        shutdown_event: Reserved for future cancellation support.

    Returns:
        Dict with:
        - stocks: top 15 by predicted weekly return (with rank)
        - total_candidates: symbols with valid predictions
        - total_universe: total halal_new symbols
        - filtered_insufficient_history: count of symbols excluded for short history
        - top_n: how many stocks were selected
        - selection_method: "patchtst_forecast"
        - model_version: PatchTST version used
        - fetched_at: ISO timestamp

    Raises:
        ValueError: If no promoted PatchTST model is available.
    """
    cached = load_cached_universe("halal_filtered")
    if cached is not None:
        return cached

    halal_new = get_halal_new_universe()
    symbols = [s["symbol"] for s in halal_new["stocks"]]
    total_universe = len(symbols)

    cutoff_date = resolve_cutoff_date()

    min_trading_days = compute_min_walkforward_days(cutoff_date)

    qualifying_symbols, excluded = filter_symbols_by_min_history(
        symbols, min_trading_days, cutoff_date
    )

    if excluded:
        for sym, days in excluded:
            logger.warning(
                f"Halal_Filtered: excluded {sym} — only {days} trading days "
                f"(need {min_trading_days})"
            )

    logger.info(
        f"Halal_Filtered: {len(qualifying_symbols)}/{total_universe} symbols pass "
        f"min-history filter ({len(excluded)} excluded), "
        f"running PatchTST inference"
    )

    batch_result = run_batch_inference(qualifying_symbols, cutoff_date)

    valid = [
        p for p in batch_result.predictions if p.predicted_weekly_return_pct is not None
    ]
    top_n = valid[:HALAL_FILTERED_TOP_N]

    logger.info(
        f"Halal_Filtered: {len(valid)} valid predictions, "
        f"returning top {len(top_n)} (model {batch_result.model_version})"
    )

    stocks = [
        {
            "symbol": p.symbol,
            "predicted_weekly_return_pct": p.predicted_weekly_return_pct,
            "rank": rank + 1,
        }
        for rank, p in enumerate(top_n)
    ]

    result = {
        "stocks": stocks,
        "total_candidates": len(valid),
        "total_universe": total_universe,
        "filtered_insufficient_history": len(excluded),
        "top_n": HALAL_FILTERED_TOP_N,
        "selection_method": "patchtst_forecast",
        "model_version": batch_result.model_version,
        "fetched_at": datetime.now(UTC).isoformat(),
    }
    save_universe_cache("halal_filtered", result)
    return result


def get_halal_filtered_symbols(
    shutdown_event: threading.Event | None = None,
) -> list[str]:
    """Get just the list of Halal_Filtered stock symbols (top 15).

    Convenience function for use by training pipelines.

    Args:
        shutdown_event: Reserved for future cancellation support.

    Returns:
        List of top 15 halal stock symbols by PatchTST predicted return.
    """
    universe = get_halal_filtered_universe(shutdown_event=shutdown_event)
    return [s["symbol"] for s in universe["stocks"]]

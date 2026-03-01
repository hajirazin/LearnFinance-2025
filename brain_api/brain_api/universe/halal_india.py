"""Halal_India stock universe: top 15 factor-scored Nifty 500 Shariah stocks.

Pipeline:
    1. Fetch Nifty 500 Shariah constituents via NSE India JSON API (~100-150 stocks)
    2. Append .NS suffix for yfinance compatibility
    3. Fetch yfinance metrics (fundamentals + 6-month price history)
    4. Attach metrics to holdings (no junk filter -- Shariah index already screens)
    5. Compute factor scores (0.4*Momentum + 0.3*Quality + 0.3*Value)
    6. Strip .NS suffix from symbols for clean display
    7. Return top 15 by factor score
"""

import logging
import threading
from datetime import UTC, datetime

from brain_api.universe.cache import load_cached_universe, save_universe_cache
from brain_api.universe.scrapers.nse import scrape_nifty500_shariah
from brain_api.universe.stock_filter import (
    compute_factor_scores,
    fetch_stock_metrics,
)

logger = logging.getLogger(__name__)

HALAL_INDIA_TOP_N = 15
NS_SUFFIX = ".NS"


def get_halal_india_universe(
    shutdown_event: threading.Event | None = None,
) -> dict:
    """Build scored India halal universe (top 15 from Nifty 500 Shariah).

    Skips junk filter because the Nifty 500 Shariah index already provides
    quality screening (Shariah compliance filters out highly leveraged and
    non-compliant companies).

    Args:
        shutdown_event: If set, aborts the yfinance fetch early.

    Returns:
        Dict with:
        - stocks: top 15 factor-scored holdings (with metrics, score, components)
        - source: "nifty_500_shariah"
        - symbol_suffix: ".NS" (signals these are NSE symbols)
        - total_stocks: count from Shariah index
        - total_scored: count with non-None factor scores
        - top_n: how many stocks were selected
        - fetched_at: ISO timestamp
    """
    cached = load_cached_universe("halal_india")
    if cached is not None:
        return cached

    nse_constituents = scrape_nifty500_shariah()
    logger.info(
        f"Halal_India: starting with {len(nse_constituents)} stocks "
        f"from Nifty 500 Shariah"
    )

    ns_holdings = [{**h, "symbol": h["symbol"] + NS_SUFFIX} for h in nse_constituents]
    ns_symbols = [h["symbol"] for h in ns_holdings]

    metrics = fetch_stock_metrics(ns_symbols, shutdown_event=shutdown_event)

    with_metrics = sum(1 for m in metrics.values() if m)
    logger.info(
        f"Halal_India: metrics coverage {with_metrics}/{len(ns_symbols)} stocks"
    )

    holdings_with_metrics = [
        {**h, "metrics": metrics.get(h["symbol"], {})} for h in ns_holdings
    ]

    scored = compute_factor_scores(holdings_with_metrics)

    scored_with_values = [s for s in scored if s.get("factor_score") is not None]
    scored_none = len(scored) - len(scored_with_values)
    if scored_none:
        logger.info(
            f"Halal_India: {scored_none} stocks scored None (missing factor data), excluded"
        )

    top_n = scored_with_values[:HALAL_INDIA_TOP_N]

    for stock in top_n:
        stock["symbol"] = stock["symbol"].removesuffix(NS_SUFFIX)

    final_symbols = [s["symbol"] for s in top_n]
    score_range = ""
    if top_n:
        best = top_n[0].get("factor_score")
        worst = top_n[-1].get("factor_score")
        score_range = f" (scores: {best:.4f} -> {worst:.4f})"
    logger.info(
        f"Halal_India: returning top {len(top_n)} of "
        f"{len(scored_with_values)} scored stocks{score_range}: {final_symbols}"
    )

    result = {
        "stocks": top_n,
        "source": "nifty_500_shariah",
        "symbol_suffix": NS_SUFFIX,
        "total_stocks": len(nse_constituents),
        "total_scored": len(scored_with_values),
        "top_n": HALAL_INDIA_TOP_N,
        "fetched_at": datetime.now(UTC).isoformat(),
    }
    save_universe_cache("halal_india", result)
    return result


def get_halal_india_symbols(
    shutdown_event: threading.Event | None = None,
) -> list[str]:
    """Get just the list of Halal_India stock symbols (top 15, clean NSE symbols).

    Args:
        shutdown_event: If set, aborts the yfinance fetch early.

    Returns:
        List of top 15 factor-scored Nifty 500 Shariah stock symbols (no .NS suffix).
    """
    universe = get_halal_india_universe(shutdown_event=shutdown_event)
    return [s["symbol"] for s in universe["stocks"]]

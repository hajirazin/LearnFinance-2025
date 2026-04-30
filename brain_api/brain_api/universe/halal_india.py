"""Halal_India stock universe: rank-band sticky top 15 on India PatchTST.

Pipeline:
    1. Get NiftyShariah500 base universe (~210 .NS-suffixed stocks)
    2. Pre-filter: exclude symbols without enough price history for
       walk-forward training (threshold derived from training window config)
    3. Run India PatchTST batch inference on qualifying symbols (OHLCV
       only; ``PatchTSTIndiaModelStorage``)
    4. Drop predictions with no valid return
    5. Apply rank-band sticky selection (K_in=15, K_hold=30) against the
       previous monthly round (partition ``halal_india_filtered_alpha``
       in the sibling ``screening_history`` table)
    6. Return the resulting 15-name slate, persisting the full candidate
       round for the next month's stickiness lookup

This produces the same count (15) as the prior blanket-top-15 selector,
keeping all downstream consumers (India PatchTST training summary email,
``/universe/halal_india`` operators) backward compatible. PatchTST India
itself is trained on the FULL ``nifty_shariah_500`` -- it never depends
on this universe -- so this rank-band sticky path is independent of the
training pipeline.

.NS suffix invariant
--------------------
``nifty_shariah_500`` produces yfinance-ready ``.NS``-suffixed symbols.
This builder preserves that suffix end-to-end -- in ``current_scores``
keys, in ``screening_history.stock`` rows, in
``evicted_from_previous`` dict keys, and in the final ``stocks`` list.
No append/strip transformations are performed.

Cadence
-------
The universe cache is **monthly** (one
``data/cache/universe/halal_india_<YYYY-MM>.json`` per calendar month).
Sticky selection therefore fires at most once per month, at the first
cache miss in a new calendar month. Within the same month, repeated
calls hit the cache and never re-run rank-band -- which is the source
of cadence isolation from the weekly India Alpha-HRP
(``halal_india_alpha`` partition in the OTHER table
``stage1_weight_history``).

Period key
----------
``screening_history.period_key`` for this builder is the ISO ``YYYYWW``
of the calendar month's first Monday (helper
``iso_year_week_of_month_anchor``). Every rebuild inside the same
calendar month therefore writes to the same ``(partition, period_key)``
bucket -- correctly handled by the screening repo's delete-then-insert
rerun rule.

Run identifier
--------------
``run_id = "universe:halal_india:<cutoff_date.isoformat()>"`` -- unique
per build attempt and idempotent under same-month rerun (because
delete-then-insert is keyed on ``(partition, period_key)``, not on
``run_id``).

Error surfaces (no silent fallbacks per AGENTS.md)
--------------------------------------------------
- Empty ``current_scores`` (India PatchTST returned nothing valid):
  ``select_with_rank_band`` raises ``ValueError`` -- propagated.
- Non-finite scores: ``rank_by_score`` (in the orchestrator) raises
  ``ValueError`` -- propagated.
- ``NseFetchError`` from the upstream ``nifty_shariah_500`` fetch
  propagates unchanged; the route handler turns it into HTTP 503.
- SQLite contention: handled by the SQLite driver's default
  serialization; concurrent ``cache miss`` builds are deterministic
  because the selector is deterministic for a fixed score map and the
  delete-then-insert is last-writer-wins for the same period bucket.

Rollout / rollback runbook
--------------------------
- First deploy: the ``halal_india_filtered_alpha`` partition in
  ``screening_history`` is empty. The first cache miss inside a calendar
  month performs a cold start -- byte-equivalent to the legacy blanket
  top-15 (under unique scores). Subsequent same-month calls hit the
  cache and never re-run rank-band. The first warm-start does not
  trigger until the next calendar month, which is the desired soft
  rollout.
- Rollback: revert this PR, delete the most recent
  ``data/cache/universe/halal_india_<YYYY-MM>.json`` so the next call
  rebuilds with the legacy blanket-top-15 logic, and (optionally) clear
  ``halal_india_filtered_alpha`` rows from
  ``data/allocation/sticky_history.db`` to remove sticky carry-set
  history (no schema change required).
- DB migration: none. Both sticky tables use ``CREATE TABLE IF NOT
  EXISTS``; the ``screening_history`` sibling table is created
  idempotently the first time ``ScreeningHistoryRepository()`` is
  instantiated (already done by the US halal_filtered builder).
- Month-rollover edge: Sunday training in month N writes screening rows
  at period_key N; Monday operations in the same month hit the cache
  (same 15). A Sunday-Monday pair spanning a calendar month boundary
  was already an edge case before this change; rank-band sticky
  narrows the divergence rather than widening it.

Out of scope (unaffected by this change)
----------------------------------------
- ``IndiaWeeklyAllocationWorkflow`` (Mon Alpha-HRP) reads
  ``nifty_shariah_500`` directly and writes to the
  ``halal_india_alpha`` partition in ``stage1_weight_history``. Zero
  overlap with this builder.
- India PatchTST training (``brain_api/routes/training/patchtst_india.py``)
  trains on ``nifty_shariah_500``, not on ``halal_india``.
- No SAC and no ETL pipeline for India in this repo.
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
from brain_api.core.screening_orchestration import select_rank_band_for_screening
from brain_api.core.sticky_selection import iso_year_week_of_month_anchor
from brain_api.core.strategy_partitions import HALAL_INDIA_FILTERED_ALPHA_PARTITION
from brain_api.storage.patchtst.local import PatchTSTIndiaModelStorage
from brain_api.storage.screening_history import ScreeningHistoryRepository
from brain_api.universe.cache import load_cached_universe, save_universe_cache
from brain_api.universe.nifty_shariah_500 import get_nifty_shariah_500_universe

logger = logging.getLogger(__name__)

HALAL_INDIA_TOP_N = 15
HALAL_INDIA_HOLD_THRESHOLD = 30
NS_SUFFIX = ".NS"


def get_halal_india_universe(
    shutdown_event: threading.Event | None = None,
) -> dict:
    """Build rank-band-sticky India halal universe (top 15).

    Symbols without enough price history for walk-forward training are
    excluded before PatchTST inference. The threshold is derived
    dynamically from the training window config.

    Args:
        shutdown_event: Reserved for future cancellation support.

    Returns:
        Dict with:
        - ``stocks``: 15 entries (sticky-kept first, then fillers; each
          entry has ``symbol`` (.NS-suffixed),
          ``predicted_weekly_return_pct``, ``rank`` (1..15 within the
          chosen slate), ``selection_reason`` (``"sticky"`` |
          ``"top_rank"``)).
        - ``total_candidates``: symbols with valid predictions this round.
        - ``total_universe``: total NiftyShariah500 symbols.
        - ``filtered_insufficient_history``: count excluded for short history.
        - ``top_n``: 15 (K_in).
        - ``selection_method``: ``"patchtst_forecast_rank_band"``.
        - ``model_version``: India PatchTST version used.
        - ``symbol_suffix``: ``".NS"`` (informational; symbols already include it).
        - ``fetched_at``: ISO timestamp.
        - ``partition``: ``"halal_india_filtered_alpha"``.
        - ``period_key``: ISO YYYYWW anchored to the month's first Monday.
        - ``previous_period_key_used``: prior period_key if warm-start, else None.
        - ``kept_count`` / ``fillers_count``: from rank-band selection.
        - ``evicted_from_previous``: ``dict[str, str]`` mapping evicted
          symbol (.NS-suffixed) -> reason.
        - ``k_in`` / ``k_hold``: 15 / 30.

    Raises:
        ValueError: If no promoted India PatchTST model is available, if
            India PatchTST inference produced no valid predictions this
            round, or if any predicted score is non-finite. No silent
            fallback to the legacy blanket-top-15 -- a broken screen is
            a hard error.
        NseFetchError: From the upstream ``nifty_shariah_500`` fetch.
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

    current_scores: dict[str, float] = {
        p.symbol: p.predicted_weekly_return_pct for p in valid
    }

    period_key = iso_year_week_of_month_anchor(cutoff_date)
    run_id = f"universe:halal_india:{cutoff_date.isoformat()}"

    repo = ScreeningHistoryRepository()
    result, previous_period_key_used = select_rank_band_for_screening(
        repo=repo,
        partition=HALAL_INDIA_FILTERED_ALPHA_PARTITION,
        period_key=period_key,
        as_of_date=cutoff_date.isoformat(),
        run_id=run_id,
        current_scores=current_scores,
        top_n=HALAL_INDIA_TOP_N,
        hold_threshold=HALAL_INDIA_HOLD_THRESHOLD,
    )

    stocks = [
        {
            "symbol": symbol,
            "predicted_weekly_return_pct": current_scores[symbol],
            "rank": idx + 1,
            "selection_reason": result.reasons.get(symbol, "top_rank"),
        }
        for idx, symbol in enumerate(result.selected)
    ]

    logger.info(
        "Halal_India: selected=%d kept=%d fillers=%d prev_period=%s evicted=%d "
        "(model %s, partition %s, period_key %s)",
        len(result.selected),
        result.kept_count,
        result.fillers_count,
        previous_period_key_used,
        len(result.evicted_from_previous),
        batch_result.model_version,
        HALAL_INDIA_FILTERED_ALPHA_PARTITION,
        period_key,
    )

    universe_result = {
        "stocks": stocks,
        "total_candidates": len(valid),
        "total_universe": total_universe,
        "filtered_insufficient_history": len(excluded),
        "top_n": HALAL_INDIA_TOP_N,
        "selection_method": "patchtst_forecast_rank_band",
        "model_version": batch_result.model_version,
        "symbol_suffix": NS_SUFFIX,
        "fetched_at": datetime.now(UTC).isoformat(),
        "partition": HALAL_INDIA_FILTERED_ALPHA_PARTITION,
        "period_key": period_key,
        "previous_period_key_used": previous_period_key_used,
        "kept_count": result.kept_count,
        "fillers_count": result.fillers_count,
        "evicted_from_previous": dict(result.evicted_from_previous),
        "k_in": HALAL_INDIA_TOP_N,
        "k_hold": HALAL_INDIA_HOLD_THRESHOLD,
    }
    save_universe_cache("halal_india", universe_result)
    return universe_result


def get_halal_india_symbols(
    shutdown_event: threading.Event | None = None,
) -> list[str]:
    """Get just the list of Halal_India stock symbols (top 15, .NS-suffixed).

    Args:
        shutdown_event: Reserved for future cancellation support.

    Returns:
        List of top 15 India stock symbols by India PatchTST predicted
        return after rank-band sticky selection (with ``.NS`` suffix,
        e.g. ``'RELIANCE.NS'``).
    """
    universe = get_halal_india_universe(shutdown_event=shutdown_event)
    return [s["symbol"] for s in universe["stocks"]]

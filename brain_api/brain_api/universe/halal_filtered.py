"""Halal_Filtered stock universe: rank-band sticky top 15 on PatchTST.

Pipeline:
    1. Get halal_new base universe (~410 stocks from 5 ETFs + Alpaca filter)
    2. Pre-filter: exclude symbols without enough price history for
       walk-forward training (threshold derived from training window config)
    3. Run PatchTST batch inference on qualifying symbols (OHLCV only)
    4. Drop predictions with no valid return
    5. Apply rank-band sticky selection (K_in=15, K_hold=30) against the
       previous monthly round (partition ``halal_filtered_alpha`` in the
       sibling ``screening_history`` table)
    6. Return the resulting 15-name slate, persisting the full candidate
       round for the next month's stickiness lookup

This produces the same count (15) as the prior blanket-top-15 selector,
keeping all downstream consumers (LSTM/PatchTST/SAC training, ETL refresh,
SAC-Monday active-symbols) backward compatible.

Cadence
-------
The universe cache is **monthly** (one ``data/cache/universe/halal_filtered_<YYYY-MM>.json``
per calendar month). Sticky selection therefore fires at most once per
month, at the first cache miss in a new calendar month. Within the same
month, repeated calls hit the cache and never re-run rank-band -- which
is the source of cadence isolation from the weekly US Alpha-HRP
(``halal_new_alpha`` partition).

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
``run_id = "universe:halal_filtered:<cutoff_date.isoformat()>"`` --
unique per build attempt and idempotent under same-month rerun (because
delete-then-insert is keyed on ``(partition, period_key)``, not on
``run_id``).

Error surfaces (no silent fallbacks per AGENTS.md)
-------------------------------------------------
- Empty ``current_scores`` (PatchTST returned nothing valid):
  ``select_with_rank_band`` raises ``ValueError`` -- propagated.
- Non-finite scores: ``rank_by_score`` (in the orchestrator) raises
  ``ValueError`` -- propagated.
- SQLite contention: handled by the SQLite driver's default
  serialization; concurrent ``cache miss`` builds are deterministic
  because the selector is deterministic for a fixed score map and the
  delete-then-insert is last-writer-wins for the same period bucket.

Rollout / rollback runbook
--------------------------
- First deploy: the ``halal_filtered_alpha`` partition in
  ``screening_history`` is empty. The first cache miss inside a calendar
  month performs a cold start -- byte-equivalent to the legacy blanket
  top-15 (under unique scores). Subsequent same-month calls hit the
  cache and never re-run rank-band. The first warm-start does not
  trigger until the next calendar month, which is the desired soft
  rollout.
- Rollback: revert this PR, delete the most recent
  ``data/cache/universe/halal_filtered_<YYYY-MM>.json`` so the next
  call rebuilds with the legacy blanket-top-15 logic, and (optionally)
  clear ``halal_filtered_alpha`` rows from
  ``data/allocation/sticky_history.db`` to remove sticky carry-set
  history (no schema change required).
- DB migration: none. Both tables use ``CREATE TABLE IF NOT EXISTS``;
  the sibling ``screening_history`` table appears next to
  ``stage1_weight_history`` in the existing DB file the first time
  ``ScreeningHistoryRepository()`` is instantiated.
- Month-rollover edge: Sunday training in month N writes Stage 1 rows
  at period_key N; Monday inference in the same month hits the cache
  (same 15). A Sunday-Monday pair spanning a calendar month boundary
  (e.g. May 31 -> Jun 1) was already an edge case before this change;
  rank-band sticky narrows the divergence rather than widening it.
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
from brain_api.core.strategy_partitions import HALAL_FILTERED_ALPHA_PARTITION
from brain_api.storage.screening_history import ScreeningHistoryRepository
from brain_api.universe.cache import load_cached_universe, save_universe_cache
from brain_api.universe.halal_new import get_halal_new_universe

logger = logging.getLogger(__name__)

HALAL_FILTERED_TOP_N = 15
HALAL_FILTERED_HOLD_THRESHOLD = 30


def get_halal_filtered_universe(
    shutdown_event: threading.Event | None = None,
) -> dict:
    """Build rank-band-sticky halal universe (top 15).

    Symbols without enough price history for walk-forward training are
    excluded before PatchTST inference. The threshold is derived
    dynamically from the training window config (training span + LSTM
    lookback buffer).

    Args:
        shutdown_event: Reserved for future cancellation support.

    Returns:
        Dict with:
        - ``stocks``: 15 entries (sticky-kept first, then fillers; each
          entry has ``symbol``, ``predicted_weekly_return_pct``,
          ``rank`` (1..15 within the chosen slate), ``selection_reason``
          (``"sticky"`` | ``"top_rank"``)).
        - ``total_candidates``: symbols with valid predictions this round.
        - ``total_universe``: total halal_new symbols.
        - ``filtered_insufficient_history``: count excluded for short history.
        - ``top_n``: 15 (K_in).
        - ``selection_method``: ``"patchtst_forecast_rank_band"``.
        - ``model_version``: PatchTST version used.
        - ``fetched_at``: ISO timestamp.
        - ``partition``: ``"halal_filtered_alpha"``.
        - ``period_key``: ISO YYYYWW anchored to the month's first Monday.
        - ``previous_period_key_used``: prior period_key if warm-start, else None.
        - ``kept_count`` / ``fillers_count``: from rank-band selection.
        - ``evicted_from_previous``: list of ``[symbol, reason]`` pairs.
        - ``k_in`` / ``k_hold``: 15 / 30.

    Raises:
        ValueError: If PatchTST inference produced no valid predictions
            this round, or if any predicted score is non-finite. No
            silent fallback to the legacy blanket-top-15 -- a broken
            screen is a hard error.
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

    current_scores: dict[str, float] = {
        p.symbol: p.predicted_weekly_return_pct for p in valid
    }

    period_key = iso_year_week_of_month_anchor(cutoff_date)
    run_id = f"universe:halal_filtered:{cutoff_date.isoformat()}"

    repo = ScreeningHistoryRepository()
    result, previous_period_key_used = select_rank_band_for_screening(
        repo=repo,
        partition=HALAL_FILTERED_ALPHA_PARTITION,
        period_key=period_key,
        as_of_date=cutoff_date.isoformat(),
        run_id=run_id,
        current_scores=current_scores,
        top_n=HALAL_FILTERED_TOP_N,
        hold_threshold=HALAL_FILTERED_HOLD_THRESHOLD,
    )

    score_by_symbol = current_scores
    stocks = [
        {
            "symbol": symbol,
            "predicted_weekly_return_pct": score_by_symbol[symbol],
            "rank": idx + 1,
            "selection_reason": result.reasons.get(symbol, "top_rank"),
        }
        for idx, symbol in enumerate(result.selected)
    ]

    logger.info(
        "Halal_Filtered: selected=%d kept=%d fillers=%d prev_period=%s evicted=%d "
        "(model %s, partition %s, period_key %s)",
        len(result.selected),
        result.kept_count,
        result.fillers_count,
        previous_period_key_used,
        len(result.evicted_from_previous),
        batch_result.model_version,
        HALAL_FILTERED_ALPHA_PARTITION,
        period_key,
    )

    universe_result = {
        "stocks": stocks,
        "total_candidates": len(valid),
        "total_universe": total_universe,
        "filtered_insufficient_history": len(excluded),
        "top_n": HALAL_FILTERED_TOP_N,
        "selection_method": "patchtst_forecast_rank_band",
        "model_version": batch_result.model_version,
        "fetched_at": datetime.now(UTC).isoformat(),
        "partition": HALAL_FILTERED_ALPHA_PARTITION,
        "period_key": period_key,
        "previous_period_key_used": previous_period_key_used,
        "kept_count": result.kept_count,
        "fillers_count": result.fillers_count,
        "evicted_from_previous": dict(result.evicted_from_previous),
        "k_in": HALAL_FILTERED_TOP_N,
        "k_hold": HALAL_FILTERED_HOLD_THRESHOLD,
    }
    save_universe_cache("halal_filtered", universe_result)
    return universe_result


def get_halal_filtered_symbols(
    shutdown_event: threading.Event | None = None,
) -> list[str]:
    """Get just the list of Halal_Filtered stock symbols (top 15).

    Convenience function for use by training pipelines.

    Args:
        shutdown_event: Reserved for future cancellation support.

    Returns:
        List of top 15 halal stock symbols (rank-band sticky selection).
    """
    universe = get_halal_filtered_universe(shutdown_event=shutdown_event)
    return [s["symbol"] for s in universe["stocks"]]

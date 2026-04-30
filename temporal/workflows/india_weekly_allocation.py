"""India Alpha-HRP weekly allocation workflow.

Pipeline (Monday weekly, 09:00 IST). Mirrors :class:`USAlphaHRPWorkflow`
phase-for-phase except India does not trade through Alpaca, so the
``resolve_next_attempt`` / ``get_hrp_portfolio`` / ``generate_orders``
/ ``sell_wait_buy`` / ``submit_orders`` phases are absent. The
remaining phases are the same Alpha-HRP pipeline:

0. Fetch the full Nifty Shariah 500 universe (~210 symbols).
1. Phase 1: PatchTST batch inference on the full universe ->
   ``{symbol -> predicted_weekly_return_pct}`` (alpha screen). Uses the
   India-trained PatchTST artifacts (``data/models/patchtst_india/``)
   via the ``score_halal_india_with_patchtst`` activity which targets
   ``POST /inference/patchtst/score-batch`` with ``market='india'``.
1.5. Phase 1.5: Rank-band sticky selection
   (``universe='halal_india_alpha'``, K_in=15, K_hold=30) -> 15
   chosen Indian symbols. Persists score rows in sticky_history.db
   under the ``halal_india_alpha`` partition so next week's selection
   has a basis for stickiness. Note the partition is distinct from
   the US ``halal_new_alpha`` partition by mathematical requirement;
   conflating sticky carry-sets across markets corrupts the
   "previously-held" signal.
2. Phase 2: HRP allocation (``lookback_days=252``) on the 15 chosen
   Indian symbols only -- math invariant: Stage 2 risk-parity must
   run on the exact selected set.
2.5. Phase 2.5: Record final stage 2 weights so next week's sticky
   read sees which Indian stocks were *actually* held.
3. Phase 3: LLM summary (``/llm/india-alpha-hrp-summary``).
4. Phase 4: Email report (``/email/india-alpha-hrp-report``).

The math primitives (PatchTST forward pass, rank-band selector, HRP
allocator, sticky_history persistence) are shared one-implementation
with the US path; only the trained PatchTST weights and the sticky
partition key are India-specific. See
``brain_api.core.strategy_partitions`` for the partition rationale.

Math correctness invariants:
- Stage 2 HRP runs **only** on the selected set; stickiness affects
  *which 15* go into Stage 2, never the HRP math itself.
- Rank-band selection is scale-invariant: K_in/K_hold are integer
  ranks, not score thresholds. PatchTST forecasts can be noisy at the
  cardinal level without destabilising selection.
"""

from dataclasses import dataclass
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from activities.inference import (
        allocate_hrp,
        record_final_weights,
        score_halal_india_with_patchtst,
        select_rank_band_top_n,
    )
    from activities.reporting import (
        generate_india_alpha_hrp_summary,
        send_india_alpha_hrp_email,
    )
    from activities.training import fetch_nifty_shariah_500_universe

ACTIVITY_TIMEOUT = timedelta(minutes=5)
HRP_TIMEOUT = timedelta(minutes=10)
PATCHTST_TIMEOUT = timedelta(minutes=10)
ACTIVITY_RETRY = 2


@dataclass(frozen=True)
class IndiaAlphaHrpStrategyParams:
    """Typed strategy knobs for the India Alpha-HRP weekly run.

    Mirrors :class:`UsAlphaHrpStrategyParams` -- same K_in/K_hold,
    same Stage 2 lookback. The only intentional difference is the
    sticky partition key, which MUST be distinct from the US partition
    so the "previously-held" rank signal is not cross-contaminated
    between strategies (see
    :mod:`brain_api.core.strategy_partitions`).
    """

    sticky_partition: str = "halal_india_alpha"
    stage2_lookback_days: int = 252
    top_n: int = 15  # K_in (entry threshold)
    hold_threshold: int = 30  # K_hold (sticky retention threshold)


DEFAULT_STRATEGY_PARAMS = IndiaAlphaHrpStrategyParams()


@workflow.defn
class IndiaWeeklyAllocationWorkflow:
    @workflow.run
    async def run(self) -> dict:
        params = DEFAULT_STRATEGY_PARAMS
        now_ist = workflow.now().astimezone()
        as_of_date = now_ist.strftime("%Y-%m-%d")
        # ISO year-week 'YYYYWW'; %G%V is correct across year boundaries
        # (week 1 of 2026 -> '202601' even when calendar is Dec 2025).
        year_week = now_ist.strftime("%G%V")
        run_id = f"paper:{as_of_date}"
        target_week_start = as_of_date
        target_week_end = (now_ist + timedelta(days=4)).strftime("%Y-%m-%d")

        workflow.logger.info(
            f"India Alpha-HRP: as_of={as_of_date} year_week={year_week} "
            f"week={target_week_start} -> {target_week_end}"
        )

        # Phase 0: Fetch the full Nifty Shariah 500 universe.
        # Distinct from get_halal_india_universe (which already pre-filters
        # to top-15 via PatchTST and would discard ranks 16-25 the
        # rank-band selector and email both want to see).
        universe_data = await workflow.execute_activity(
            fetch_nifty_shariah_500_universe,
            start_to_close_timeout=ACTIVITY_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
        )
        india_symbols = [s["symbol"] for s in universe_data.get("stocks", [])]
        workflow.logger.info(f"nifty_shariah_500 universe={len(india_symbols)} symbols")

        # Phase 1: PatchTST alpha scores across the full Indian universe.
        scores = await workflow.execute_activity(
            score_halal_india_with_patchtst,
            args=[india_symbols, as_of_date, params.top_n],
            start_to_close_timeout=PATCHTST_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
        )
        workflow.logger.info(
            f"Stage 1: PatchTST scored {scores.predicted_count}/"
            f"{scores.requested_count} (excluded={len(scores.excluded_symbols)}), "
            f"model={scores.model_version}"
        )

        # Phase 1.5: Rank-band sticky selection on the halal_india_alpha
        # partition (distinct from US halal_new_alpha by mathematical
        # requirement -- see brain_api.core.strategy_partitions).
        sticky = await workflow.execute_activity(
            select_rank_band_top_n,
            args=[
                scores.scores,
                params.sticky_partition,
                year_week,
                as_of_date,
                run_id,
                params.top_n,
                params.hold_threshold,
            ],
            start_to_close_timeout=ACTIVITY_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
        )
        selected_symbols = sticky.selected
        workflow.logger.info(
            f"Sticky: kept={sticky.kept_count} fillers={sticky.fillers_count} "
            f"prev_yw={sticky.previous_year_week_used} "
            f"selected={selected_symbols}"
        )

        # Phase 2: HRP on the chosen N (math runs only on the selected set).
        stage2 = await workflow.execute_activity(
            allocate_hrp,
            args=[selected_symbols, as_of_date, params.stage2_lookback_days],
            start_to_close_timeout=HRP_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
        )
        workflow.logger.info(f"Stage 2: {stage2.symbols_used} symbols allocated")

        # Phase 2.5: Record stage 2 weights so next week's sticky read
        # can see which stocks were *actually* held.
        await workflow.execute_activity(
            record_final_weights,
            args=[params.sticky_partition, year_week, stage2.percentage_weights],
            start_to_close_timeout=ACTIVITY_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
        )

        # Phase 3: LLM summary across alpha screen + Stage 2 HRP.
        summary = await workflow.execute_activity(
            generate_india_alpha_hrp_summary,
            args=[
                scores,
                sticky,
                stage2,
                params.sticky_partition,
                params.top_n,
                params.hold_threshold,
            ],
            start_to_close_timeout=ACTIVITY_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
        )

        # Phase 4: Email report. India does not trade -- no order_results.
        email = await workflow.execute_activity(
            send_india_alpha_hrp_email,
            args=[
                summary,
                scores,
                sticky,
                stage2,
                params.sticky_partition,
                params.top_n,
                params.hold_threshold,
                target_week_start,
                target_week_end,
                as_of_date,
            ],
            start_to_close_timeout=ACTIVITY_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
        )

        return {
            "as_of_date": as_of_date,
            "year_week": year_week,
            "target_week_start": target_week_start,
            "target_week_end": target_week_end,
            "universe_symbols": len(india_symbols),
            "stage1_predicted_count": scores.predicted_count,
            "stage1_excluded_count": len(scores.excluded_symbols),
            "model_version": scores.model_version,
            "top_n": params.top_n,
            "hold_threshold": params.hold_threshold,
            "selected_symbols": selected_symbols,
            "kept_count": sticky.kept_count,
            "fillers_count": sticky.fillers_count,
            "previous_year_week_used": sticky.previous_year_week_used,
            "stage2_symbols_used": stage2.symbols_used,
            "email": {
                "is_success": email.is_success,
                "subject": email.subject,
            },
        }

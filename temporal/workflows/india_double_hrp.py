"""India Double HRP workflow with weight-band sticky selection.

Two-stage Hierarchical Risk Parity over the Nifty Shariah 500 NSE
universe (~210 stocks), with a brain_api-side sticky-selection layer
that retains last week's holdings whose Stage 1 weight moved by less
than ``STICKINESS_THRESHOLD_PP`` absolute percentage points.

Phase ordering (mirrors :class:`USDoubleHRPWorkflow` minus the Alpaca
order-execution legs because India is paper-only with no broker):

0. Fetch the full Nifty Shariah 500 universe.
1. Stage 1 HRP: ``allocate_hrp(all_symbols, 756d)``.
1.5. ``select_sticky_top_n``: persists Stage 1 weights against the
     ``halal_india_double_hrp`` partition and applies the weight-band
     against last week's final 15 -> chosen 15 symbols.
2. Stage 2 HRP: ``allocate_hrp(sticky.selected, 252d)``.
2.5. ``record_final_weights``: writes Stage 2 weights back into Stage 1
     history rows so next week's sticky read sees the actually-held set.
3. ``generate_double_hrp_summary`` (LLM, with sticky context).
4. ``send_double_hrp_email`` (HTML report, with sticky context).

Math correctness invariant: Stage 2 HRP MUST run on
``sticky.selected``, not on the raw top-N by Stage 1 weight, so the
final weights are mathematically consistent with the chosen symbols.
Stickiness only affects *selection*, never the final allocation math.

Sticky-history isolation: this workflow MUST use the
``halal_india_double_hrp`` partition (see
:mod:`brain_api.core.strategy_partitions`). Reusing
``halal_india_alpha`` would corrupt the rank-band carry-set used by
:class:`IndiaWeeklyAllocationWorkflow`; reusing ``halal_new`` would
mix US carry-state into India.
"""

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from activities.email_enrichment import build_prior_allocation_from_db
    from activities.inference import (
        allocate_hrp,
        get_previous_final_allocation,
        record_final_weights,
        select_sticky_top_n,
    )
    from activities.reporting import (
        generate_double_hrp_summary,
        send_double_hrp_email,
    )
    from activities.training import fetch_nifty_shariah_500_universe

ACTIVITY_TIMEOUT = timedelta(minutes=5)
HRP_TIMEOUT = timedelta(minutes=10)
ACTIVITY_RETRY = 2

# Partition string also doubles as the email "Universe" display label
# -- mirrors :class:`IndiaWeeklyAllocationWorkflow` which shows
# ``halal_india_alpha``. The actual scrape source (Nifty Shariah 500)
# is documented in the email's "Strategy:" line and Stage 1 symbol
# count.
SELECT_PARTITION = "halal_india_double_hrp"
STAGE1_LOOKBACK = 756  # ~3 years (full universe screening)
STAGE2_LOOKBACK = 252  # ~1 year (final allocation on chosen 15)
TOP_N = 15
STICKINESS_THRESHOLD_PP = 1.0


@workflow.defn
class IndiaDoubleHRPWorkflow:
    @workflow.run
    async def run(self) -> dict:
        now_ist = workflow.now().astimezone()
        as_of_date = now_ist.strftime("%Y-%m-%d")
        # ISO year-week 'YYYYWW'; %G%V is correct across year boundaries.
        year_week = now_ist.strftime("%G%V")
        # India is paper-only with no broker, so there is no Alpaca
        # account to dedup against. We still mint a run_id with the
        # default ``paper:YYYY-MM-DD`` form for log-trace consistency
        # with the US workflows.
        run_id = f"paper:{as_of_date}"
        target_week_start = as_of_date
        target_week_end = (now_ist + timedelta(days=4)).strftime("%Y-%m-%d")

        workflow.logger.info(
            f"India Double HRP: as_of={as_of_date} year_week={year_week} "
            f"week={target_week_start} -> {target_week_end}"
        )

        # Phase 0: Fetch full Nifty Shariah 500 universe
        universe_data = await workflow.execute_activity(
            fetch_nifty_shariah_500_universe,
            start_to_close_timeout=ACTIVITY_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
        )
        all_symbols = [s["symbol"] for s in universe_data.get("stocks", [])]
        workflow.logger.info(f"Universe: {len(all_symbols)} symbols")

        # Phase 1: Stage 1 HRP across the full Nifty Shariah 500 universe.
        stage1 = await workflow.execute_activity(
            allocate_hrp,
            args=[all_symbols, as_of_date, STAGE1_LOOKBACK],
            start_to_close_timeout=HRP_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
        )
        workflow.logger.info(
            f"Stage 1: {stage1.symbols_used} allocated, "
            f"{len(stage1.symbols_excluded)} excluded"
        )

        # Phase 1.5: Weight-band sticky selection on the
        # halal_india_double_hrp partition (distinct from
        # halal_india_alpha by mathematical requirement -- different
        # selector primitive, see brain_api.core.strategy_partitions).
        sticky = await workflow.execute_activity(
            select_sticky_top_n,
            args=[
                stage1,
                SELECT_PARTITION,
                year_week,
                as_of_date,
                run_id,
                TOP_N,
                STICKINESS_THRESHOLD_PP,
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

        # Phase 2: Stage 2 HRP on the chosen N (math runs only on selected).
        stage2 = await workflow.execute_activity(
            allocate_hrp,
            args=[selected_symbols, as_of_date, STAGE2_LOOKBACK],
            start_to_close_timeout=HRP_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
        )
        workflow.logger.info(f"Stage 2: {stage2.symbols_used} symbols allocated")

        # Phase 2.5: Record final stage 2 weights so next week's sticky
        # read can see which stocks were *actually* held.
        await workflow.execute_activity(
            record_final_weights,
            args=[SELECT_PARTITION, year_week, stage2.percentage_weights],
            start_to_close_timeout=ACTIVITY_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
        )

        # Phase 3: AI summary (with sticky context so the prompt can
        # describe weight-band stability vs prior week).
        summary = await workflow.execute_activity(
            generate_double_hrp_summary,
            args=[
                stage1,
                stage2,
                SELECT_PARTITION,
                TOP_N,
                sticky.kept_count,
                sticky.fillers_count,
                sticky.previous_year_week_used,
                STICKINESS_THRESHOLD_PP,
            ],
            start_to_close_timeout=ACTIVITY_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
        )

        # Read prior week's final weights for the "Going Into This Week"
        # email block. India is paper-only so the DB row is the truth
        # (no live broker drift to worry about).
        prior_final = await workflow.execute_activity(
            get_previous_final_allocation,
            args=[SELECT_PARTITION, year_week],
            start_to_close_timeout=ACTIVITY_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
        )
        prior_allocation = build_prior_allocation_from_db(
            prior_final.final_weights_pct,
            source_label=(
                f"recorded last week ({prior_final.year_week})"
                if prior_final.year_week
                else "recorded last week (cold start)"
            ),
            as_of=prior_final.year_week,
        )

        # Phase 4: Email report (paper-only -- no order_results / skip).
        email_result = await workflow.execute_activity(
            send_double_hrp_email,
            args=[
                summary,
                stage1,
                stage2,
                SELECT_PARTITION,
                TOP_N,
                target_week_start,
                target_week_end,
                as_of_date,
                sticky.kept_count,
                sticky.fillers_count,
                sticky.previous_year_week_used,
                STICKINESS_THRESHOLD_PP,
                prior_allocation,
            ],
            start_to_close_timeout=ACTIVITY_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
        )

        return {
            "as_of_date": as_of_date,
            "year_week": year_week,
            "target_week_start": target_week_start,
            "target_week_end": target_week_end,
            "universe_symbols": len(all_symbols),
            "stage1_symbols_used": stage1.symbols_used,
            "stage1_symbols_excluded": len(stage1.symbols_excluded),
            "top_n": TOP_N,
            "selected_symbols": selected_symbols,
            "kept_count": sticky.kept_count,
            "fillers_count": sticky.fillers_count,
            "previous_year_week_used": sticky.previous_year_week_used,
            "stage2_symbols_used": stage2.symbols_used,
            "summary_provider": summary.provider,
            "email": {
                "is_success": email_result.is_success,
                "subject": email_result.subject,
            },
        }

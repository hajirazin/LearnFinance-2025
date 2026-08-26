"""US weekly allocation workflow (SAC-only) with durable sell-wait-buy.

Runs every Monday at 09:00 America/New_York.

History note (post-refactor): this workflow used to run a "naive HRP"
allocator side-by-side with SAC on SAC's 15-stock universe. That naive
HRP path has been **retired** in favor of the dedicated
``USAlphaHRPWorkflow`` (PatchTST alpha screen on the full halal_new
universe -> rank-band sticky -> HRP). The HRP Alpaca paper account is
unchanged; only the strategy that produces its weekly weights changed.

The SAC weekly LLM summary / email lives at ``/llm/sac-weekly-summary``
and ``/email/sac-weekly-report`` (renamed from the legacy
``/llm/weekly-summary`` / ``/email/weekly-report`` once HRP was
removed from this path).

Phases (SAC-only):
0. Get active symbols + SAC portfolio (parallel)
1. Get signals + forecasts (parallel)
2. Run SAC allocator (skipped if open orders on sac account)
3. Generate SAC orders + store SAC experience
4. SAC sell-wait-buy
5. Get SAC order history + update execution
6. Generate SAC LLM summary + send SAC weekly email
"""

import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

from workflows._order_execution import (
    SHORT_TIMEOUT,
    combine_order_phases,
    combine_submit,
    sell_wait_buy,
    split_orders_by_side,
    submit_sells_and_wait,
)
from workflows._run_identity import ist_calendar_date

with workflow.unsafe.imports_passed_through():
    from activities.email_enrichment import (
        build_order_details,
        build_prior_allocation_from_portfolio,
    )
    from activities.execution import (
        generate_buy_orders_sac,
        generate_orders_sac,
        generate_sell_orders_sac,
        store_experience_sac,
        update_execution_sac,
    )
    from activities.inference import (
        get_adjusted_closes,
        get_market_history,
        get_monday_decision_window,
        get_patchtst_forecast,
        infer_sac,
        materialize_news_window,
        query_news_window,
    )
    from activities.portfolio import (
        get_active_symbols,
        get_order_history_sac,
        get_sac_portfolio,
        resolve_next_attempt,
        submit_orders_sac,
    )
    from activities.reporting import (
        generate_summary,
        send_weekly_email,
    )
    from models import SkippedAllocation
    from models.news import SACNewsAudit

INFERENCE_TIMEOUT = timedelta(minutes=20)
INFER_SAC_TIMEOUT = timedelta(minutes=15)


def _news_audit_for_summary(sac_alloc, window) -> SACNewsAudit:
    audit = getattr(sac_alloc, "news_audit", None)
    if audit is not None:
        return audit
    return SACNewsAudit(
        as_of=window.cutoff,
        start_exclusive=window.start_exclusive,
        end_inclusive=window.end_inclusive,
        per_symbol=[],
    )


@workflow.defn
class USWeeklyAllocationWorkflow:
    """SAC-only US weekly allocation workflow.

    Naive HRP was retired in favor of ``USAlphaHRPWorkflow``; this class
    name is preserved to avoid downstream renames in the schedule and
    worker registry.
    """

    @workflow.run
    async def run(self) -> dict:
        as_of_date = ist_calendar_date(workflow.now())
        run_id = f"paper:{as_of_date}"

        attempt = await workflow.execute_activity(
            resolve_next_attempt,
            args=[run_id, as_of_date, ["sac"]],
            start_to_close_timeout=SHORT_TIMEOUT,
        )

        workflow.logger.info(
            f"Starting US weekly allocation pipeline (SAC-only, attempt={attempt})..."
        )

        # Phase 0: Get active symbols + SAC portfolio (parallel)
        active_symbols, sac_portfolio = await asyncio.gather(
            workflow.execute_activity(
                get_active_symbols,
                args=["halal_filtered"],
                start_to_close_timeout=SHORT_TIMEOUT,
            ),
            workflow.execute_activity(
                get_sac_portfolio, start_to_close_timeout=SHORT_TIMEOUT
            ),
        )

        symbols = active_symbols.symbols
        if not active_symbols.training_cutoff_date:
            raise RuntimeError("SAC v3 active-symbol metadata lacks training cutoff")
        if active_symbols.sac_schema_version != 3:
            raise RuntimeError("SAC active-symbol metadata is not schema version 3")
        run_sac = sac_portfolio.open_orders_count == 0

        skipped_algorithms = []
        if not run_sac:
            skipped_algorithms.append("SAC")

        decision_window = await workflow.execute_activity(
            get_monday_decision_window,
            args=[as_of_date],
            start_to_close_timeout=SHORT_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=3),
        )
        as_of = decision_window.cutoff

        # Phase 1: materialize news in parallel with price/forecast evidence,
        # then query the parse-only DTO.
        _, patchtst, prices, market = await asyncio.gather(
            workflow.execute_activity(
                materialize_news_window,
                args=[symbols, decision_window],
                start_to_close_timeout=INFERENCE_TIMEOUT,
                retry_policy=RetryPolicy(maximum_attempts=3),
            ),
            workflow.execute_activity(
                get_patchtst_forecast,
                args=[as_of_date, symbols],
                start_to_close_timeout=INFERENCE_TIMEOUT,
                retry_policy=RetryPolicy(maximum_attempts=3),
            ),
            workflow.execute_activity(
                get_adjusted_closes,
                args=[symbols, as_of_date],
                start_to_close_timeout=INFERENCE_TIMEOUT,
                retry_policy=RetryPolicy(maximum_attempts=3),
            ),
            workflow.execute_activity(
                get_market_history,
                args=[active_symbols.training_cutoff_date, as_of_date],
                start_to_close_timeout=INFERENCE_TIMEOUT,
                retry_policy=RetryPolicy(maximum_attempts=3),
            ),
        )
        news_window = await workflow.execute_activity(
            query_news_window,
            args=[symbols, decision_window],
            start_to_close_timeout=SHORT_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        target_week_start = patchtst.target_week_start or as_of_date
        target_week_end = patchtst.target_week_end or as_of_date

        # Phase 2: SAC allocator (skipped if open orders on sac account).
        if run_sac:
            sac_alloc = await workflow.execute_activity(
                infer_sac,
                args=[
                    sac_portfolio,
                    as_of_date,
                    "halal_filtered",
                    symbols,
                    as_of,
                    news_window,
                    patchtst,
                    prices,
                    market,
                ],
                start_to_close_timeout=INFER_SAC_TIMEOUT,
            )
        else:
            sac_alloc = SkippedAllocation(algorithm="sac")

        fresh_buy_pricing = workflow.patched("sac-fresh-buy-pricing-v1")
        if fresh_buy_pricing:
            # Brain prices and generates sells immediately.
            sac_sell_orders = await workflow.execute_activity(
                generate_sell_orders_sac,
                args=[sac_alloc, sac_portfolio, run_id, attempt, "sac"],
                start_to_close_timeout=SHORT_TIMEOUT,
            )
            sac_orders = sac_sell_orders
        else:
            # Replay-only path for workflows started before the phase split.
            sac_orders = await workflow.execute_activity(
                generate_orders_sac,
                args=[sac_alloc, sac_portfolio, run_id, attempt, "sac"],
                start_to_close_timeout=SHORT_TIMEOUT,
            )

        # Store experience (fire-and-forget for SAC RL).
        if run_sac:
            await workflow.execute_activity(
                store_experience_sac,
                args=[
                    run_id,
                    target_week_start,
                    target_week_end,
                    sac_alloc,
                    "halal_filtered",
                ],
                start_to_close_timeout=SHORT_TIMEOUT,
            )

        if fresh_buy_pricing:
            # Submit sells and wait. Only after that durable wait do we
            # refresh broker state and ask Brain to price/generate buys.
            sell_submit = await submit_sells_and_wait(
                "sac", sac_sell_orders, submit_orders_sac
            )
            buy_portfolio = sac_portfolio
            if run_sac:
                buy_portfolio = await workflow.execute_activity(
                    get_sac_portfolio,
                    start_to_close_timeout=SHORT_TIMEOUT,
                )
            sac_buy_orders = await workflow.execute_activity(
                generate_buy_orders_sac,
                args=[sac_alloc, buy_portfolio, run_id, attempt, "sac"],
                start_to_close_timeout=SHORT_TIMEOUT,
            )
            buy_submit = await workflow.execute_activity(
                submit_orders_sac,
                args=[sac_buy_orders],
                start_to_close_timeout=SHORT_TIMEOUT,
            )
            sac_orders = combine_order_phases(sac_sell_orders, sac_buy_orders)
            sac_submit = combine_submit(sell_submit, buy_submit)
        else:
            sac_sells, sac_buys = split_orders_by_side(sac_orders)
            sac_submit = await sell_wait_buy(
                "sac", sac_sells, sac_buys, sac_orders, submit_orders_sac
            )

        # Phase 5: Get SAC order history + post-trade portfolio (parallel),
        # then update execution. The post-trade portfolio snapshot is what
        # the labeller will read back as actual_weights, so a fresh fetch
        # AFTER sell-wait-buy is what we want.
        if run_sac:
            sac_history, sac_post_portfolio = await asyncio.gather(
                workflow.execute_activity(
                    get_order_history_sac,
                    args=[target_week_start],
                    start_to_close_timeout=SHORT_TIMEOUT,
                ),
                workflow.execute_activity(
                    get_sac_portfolio,
                    start_to_close_timeout=SHORT_TIMEOUT,
                ),
            )
            await workflow.execute_activity(
                update_execution_sac,
                args=[run_id, sac_orders, sac_history, sac_post_portfolio],
                start_to_close_timeout=SHORT_TIMEOUT,
            )

        # Phase 6: Generate summary + send email
        summary = await workflow.execute_activity(
            generate_summary,
            args=[
                patchtst,
                _news_audit_for_summary(sac_alloc, decision_window),
                sac_alloc,
                "halal_filtered",
            ],
            start_to_close_timeout=SHORT_TIMEOUT,
        )

        # Build the per-order detail rows (with ATR-based stop-loss) and
        # the "going into this week" snapshot (live SAC Alpaca account
        # state at run start). Pure helpers -- no extra activity round-trip.
        sac_order_details = build_order_details(sac_orders, sac_submit)
        sac_prior_allocation = build_prior_allocation_from_portfolio(
            sac_portfolio,
            source_label="live Alpaca account: sac",
            as_of=as_of_date,
        )

        email_result = await workflow.execute_activity(
            send_weekly_email,
            args=[
                summary,
                patchtst,
                sac_alloc,
                sac_submit,
                target_week_start,
                target_week_end,
                as_of_date,
                skipped_algorithms,
                "halal_filtered",
                sac_order_details,
                sac_prior_allocation,
            ],
            start_to_close_timeout=SHORT_TIMEOUT,
        )

        workflow.logger.info("Weekly allocation pipeline complete!")

        return {
            "run_id": run_id,
            "as_of_date": as_of_date,
            "symbols_count": len(symbols),
            "skipped_algorithms": skipped_algorithms,
            "sac": {
                "orders_submitted": getattr(sac_submit, "orders_submitted", 0),
                "skipped": not run_sac,
            },
            "email": {
                "is_success": email_result.is_success,
                "subject": email_result.subject,
            },
        }

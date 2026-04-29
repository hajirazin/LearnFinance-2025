"""US Double HRP workflow with sticky selection + Alpaca paper trading.

Two-stage Hierarchical Risk Parity over the ``halal_new`` US universe
(~410 Alpaca-tradable halal symbols), with a brain_api-side sticky-
selection layer that retains last week's holdings whose Stage 1 weight
moved by less than ``STICKINESS_THRESHOLD_PP`` absolute percentage
points. Trades through a dedicated ``dhrp`` Alpaca paper account.

Phase ordering:
0. Resolve attempt + fetch halal_new universe + get DHRP portfolio (parallel).
   - If open orders > 0 -> short-circuit to skip path (email skipped=True).
1. Stage 1 HRP: allocate_hrp(all_symbols, 756d).
1.5. select_sticky_top_n: persists Stage 1 weights and applies sticky
     band against last week's final 15 -> chosen 15 symbols.
2. Stage 2 HRP: allocate_hrp(selected_15, 252d).
2.5. record_final_weights: writes Stage 2 weights back into stage 1
     history rows so next week's sticky read sees the final set.
3. generate_orders_dhrp on stage2 + dhrp_portfolio.
4. Reused sell-wait-buy machinery against the dhrp account.
5. generate_us_double_hrp_summary (LLM).
6. send_us_double_hrp_email (HTML report).

Math correctness invariant: Stage 2 HRP **must** run on the
sticky-selected set (not the raw top-N), so the final weights are
mathematically consistent with the chosen symbols. Stickiness only
affects *selection*, never the final allocation math.
"""

import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

from workflows._order_execution import (
    SHORT_TIMEOUT,
    sell_wait_buy,
    split_orders_by_side,
)

with workflow.unsafe.imports_passed_through():
    from activities.execution import generate_orders_dhrp
    from activities.inference import (
        allocate_hrp,
        record_final_weights,
        select_sticky_top_n,
    )
    from activities.portfolio import (
        get_dhrp_portfolio,
        resolve_next_attempt,
        submit_orders_dhrp,
    )
    from activities.reporting import (
        generate_us_double_hrp_summary,
        send_us_double_hrp_email,
    )
    from activities.training import fetch_halal_new_universe
    from models import HRPAllocationResponse, SkippedSubmitResponse

ACTIVITY_TIMEOUT = timedelta(minutes=5)
HRP_TIMEOUT = timedelta(minutes=10)
ACTIVITY_RETRY = 2

UNIVERSE_LABEL = "halal_new"
STAGE1_LOOKBACK = 756  # ~3 years (full universe screening)
STAGE2_LOOKBACK = 252  # ~1 year (final allocation on chosen 15)
TOP_N = 15
STICKINESS_THRESHOLD_PP = 1.0


@workflow.defn
class USDoubleHRPWorkflow:
    @workflow.run
    async def run(self) -> dict:
        now_ist = workflow.now().astimezone()
        as_of_date = now_ist.strftime("%Y-%m-%d")
        # ISO year-week ('YYYYWW'), e.g. '202609'. %G%V is correct across
        # year boundaries (week 1 of 2026 -> '202601' even if calendar is
        # Dec 2025).
        year_week = now_ist.strftime("%G%V")
        run_id = f"paper:{as_of_date}"
        target_week_start = as_of_date
        target_week_end = (now_ist + timedelta(days=4)).strftime("%Y-%m-%d")

        workflow.logger.info(
            f"US Double HRP: as_of={as_of_date} year_week={year_week} "
            f"week={target_week_start} -> {target_week_end}"
        )

        # Phase 0: attempt + universe + portfolio (parallel)
        attempt, universe_data, dhrp_portfolio = await asyncio.gather(
            workflow.execute_activity(
                resolve_next_attempt,
                args=[run_id, as_of_date, ["dhrp"]],
                start_to_close_timeout=ACTIVITY_TIMEOUT,
            ),
            workflow.execute_activity(
                fetch_halal_new_universe,
                start_to_close_timeout=ACTIVITY_TIMEOUT,
                retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
            ),
            workflow.execute_activity(
                get_dhrp_portfolio,
                start_to_close_timeout=ACTIVITY_TIMEOUT,
            ),
        )

        halal_new_symbols = [s["symbol"] for s in universe_data.get("stocks", [])]
        workflow.logger.info(
            f"halal_new universe={len(halal_new_symbols)} symbols, "
            f"dhrp_portfolio open_orders={dhrp_portfolio.open_orders_count}, "
            f"attempt={attempt}"
        )

        # Skip path: open orders on the DHRP account from a prior run mean
        # we cannot rebalance without breaking idempotency. Send a skipped
        # email and return early.
        #
        # Why empty HRPAllocationResponse, not SkippedAllocation:
        # the brain_api LLM/email endpoints expect ``HRPAllocationResponse``
        # for stage1/stage2. Passing a SkippedAllocation would 422. The
        # ``skipped=True`` flag in the email payload tells the template to
        # hide the allocation tables, so empty placeholders are sufficient.
        if dhrp_portfolio.open_orders_count > 0:
            workflow.logger.warning(
                f"DHRP skipped: {dhrp_portfolio.open_orders_count} open orders"
            )
            empty_alloc = HRPAllocationResponse(
                percentage_weights={},
                symbols_used=0,
                symbols_excluded=[],
                lookback_days=0,
                as_of_date=as_of_date,
            )
            summary = await workflow.execute_activity(
                generate_us_double_hrp_summary,
                args=[empty_alloc, empty_alloc, UNIVERSE_LABEL, TOP_N],
                start_to_close_timeout=ACTIVITY_TIMEOUT,
                retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
            )
            email = await workflow.execute_activity(
                send_us_double_hrp_email,
                args=[
                    summary,
                    empty_alloc,
                    empty_alloc,
                    UNIVERSE_LABEL,
                    TOP_N,
                    target_week_start,
                    target_week_end,
                    as_of_date,
                    0,
                    0,
                    None,
                    SkippedSubmitResponse(account="dhrp", skipped=True),
                    True,
                ],
                start_to_close_timeout=ACTIVITY_TIMEOUT,
                retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
            )
            return {
                "as_of_date": as_of_date,
                "year_week": year_week,
                "skipped": True,
                "skip_reason": "open_orders",
                "email": {
                    "is_success": email.is_success,
                    "subject": email.subject,
                },
            }

        # Phase 1: Stage 1 HRP across the full halal_new universe.
        stage1 = await workflow.execute_activity(
            allocate_hrp,
            args=[halal_new_symbols, as_of_date, STAGE1_LOOKBACK],
            start_to_close_timeout=HRP_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
        )
        workflow.logger.info(
            f"Stage 1: {stage1.symbols_used} allocated, "
            f"{len(stage1.symbols_excluded)} excluded"
        )

        # Phase 1.5: Sticky selection (persists stage 1 weights server-side).
        sticky = await workflow.execute_activity(
            select_sticky_top_n,
            args=[
                stage1,
                UNIVERSE_LABEL,
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
            args=[UNIVERSE_LABEL, year_week, stage2.percentage_weights],
            start_to_close_timeout=ACTIVITY_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
        )

        # Phase 3: Translate weights -> orders for the dhrp Alpaca account.
        orders = await workflow.execute_activity(
            generate_orders_dhrp,
            args=[stage2, dhrp_portfolio, run_id, attempt],
            start_to_close_timeout=SHORT_TIMEOUT,
        )

        # Phase 4: Sell -> wait for terminal -> buy. Reused helper keeps
        # the durable polling loop in a single place.
        sells, buys = split_orders_by_side(orders)
        submit = await sell_wait_buy("dhrp", sells, buys, orders, submit_orders_dhrp)

        # Phase 5: LLM summary on the two stages.
        summary = await workflow.execute_activity(
            generate_us_double_hrp_summary,
            args=[stage1, stage2, UNIVERSE_LABEL, TOP_N],
            start_to_close_timeout=ACTIVITY_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
        )

        # Phase 6: Email report.
        email = await workflow.execute_activity(
            send_us_double_hrp_email,
            args=[
                summary,
                stage1,
                stage2,
                UNIVERSE_LABEL,
                TOP_N,
                target_week_start,
                target_week_end,
                as_of_date,
                sticky.kept_count,
                sticky.fillers_count,
                sticky.previous_year_week_used,
                submit,
                False,
            ],
            start_to_close_timeout=ACTIVITY_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
        )

        if isinstance(submit, SkippedSubmitResponse):
            orders_submitted = 0
            orders_failed = 0
        else:
            orders_submitted = submit.orders_submitted
            orders_failed = submit.orders_failed

        return {
            "as_of_date": as_of_date,
            "year_week": year_week,
            "target_week_start": target_week_start,
            "target_week_end": target_week_end,
            "universe_symbols": len(halal_new_symbols),
            "stage1_symbols_used": stage1.symbols_used,
            "stage1_symbols_excluded": len(stage1.symbols_excluded),
            "top_n": TOP_N,
            "selected_symbols": selected_symbols,
            "kept_count": sticky.kept_count,
            "fillers_count": sticky.fillers_count,
            "previous_year_week_used": sticky.previous_year_week_used,
            "stage2_symbols_used": stage2.symbols_used,
            "orders_submitted": orders_submitted,
            "orders_failed": orders_failed,
            "skipped": False,
            "email": {
                "is_success": email.is_success,
                "subject": email.subject,
            },
        }

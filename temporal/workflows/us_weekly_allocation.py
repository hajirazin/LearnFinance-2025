"""US weekly allocation workflow with durable sell-wait-buy.

Runs every Monday at 18:00 IST (11:00 UTC). Single workflow replaces the
3-flow Prefect hack (sells flow + monitor cron + buys flow).

Phases:
0. Get active symbols + Alpaca portfolios (parallel)
1. Get signals + forecasts (parallel)
2. Run allocators: SAC, HRP (parallel, conditional on open orders)
3. Generate orders + store experience
4. Per-algorithm sell-wait-buy (parallel per account, each independently
   submits sells -> polls until terminal -> submits buys)
5. Get order history + update execution
6. Generate LLM summary + send email
"""

import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from activities.execution import (
        generate_orders_hrp,
        generate_orders_sac,
        store_experience_sac,
        update_execution_sac,
    )
    from activities.inference import (
        allocate_hrp,
        get_fundamentals,
        get_lstm_forecast,
        get_news_sentiment,
        get_patchtst_forecast,
        infer_sac,
    )
    from activities.portfolio import (
        check_order_statuses,
        get_active_symbols,
        get_hrp_portfolio,
        get_order_history_sac,
        get_sac_portfolio,
        resolve_next_attempt,
        submit_orders_hrp,
        submit_orders_sac,
    )
    from activities.reporting import generate_summary, send_weekly_email
    from models import (
        GenerateOrdersResponse,
        OrderModel,
        SkippedAllocation,
        SkippedOrdersResponse,
        SkippedSubmitResponse,
    )

SHORT_TIMEOUT = timedelta(minutes=5)
# Must be greater than the httpx read timeout in temporal/activities/client.py
# (currently 15 min) so httpx times out before Temporal does, allowing clean
# retries. Pi FinBERT sentiment is the slowest activity at 5-6 min.
INFERENCE_TIMEOUT = timedelta(minutes=20)
SELL_POLL_INTERVAL = timedelta(minutes=15)
SELL_DEADLINE = timedelta(hours=48)

TERMINAL_STATUSES = {"filled", "canceled", "expired", "rejected", "replaced"}


def _split_orders_by_side(
    orders_resp: GenerateOrdersResponse | SkippedOrdersResponse,
) -> tuple[GenerateOrdersResponse | SkippedOrdersResponse, list[OrderModel]]:
    """Split orders into sell-only response and buy order list.

    Returns (sell_only_response, buy_orders_list).
    """
    if isinstance(orders_resp, SkippedOrdersResponse) or getattr(
        orders_resp, "skipped", False
    ):
        return orders_resp, []

    sell_orders = [o for o in orders_resp.orders if o.side == "sell"]
    buy_orders = [o for o in orders_resp.orders if o.side == "buy"]

    sell_response = GenerateOrdersResponse(
        orders=sell_orders,
        summary=orders_resp.summary,
        prices_used=orders_resp.prices_used,
    )
    return sell_response, buy_orders


def _make_buy_response(
    buy_orders: list[OrderModel],
    original: GenerateOrdersResponse | SkippedOrdersResponse,
) -> GenerateOrdersResponse | SkippedOrdersResponse:
    """Reconstruct a GenerateOrdersResponse with buy-only orders."""
    if isinstance(original, SkippedOrdersResponse):
        return original
    return GenerateOrdersResponse(
        orders=buy_orders,
        summary=original.summary,
        prices_used=original.prices_used,
    )


async def _sell_wait_buy(
    account: str,
    sells: GenerateOrdersResponse | SkippedOrdersResponse,
    buy_orders: list[OrderModel],
    original_orders: GenerateOrdersResponse | SkippedOrdersResponse,
    submit_activity,
):
    """Run the full sell -> poll -> buy cycle for a single algorithm.

    Each algorithm (SAC/HRP) has its own Alpaca account, so their
    sell-wait-buy pipelines are fully independent and run in parallel.
    """
    sell_submit = await workflow.execute_activity(
        submit_activity,
        args=[sells],
        start_to_close_timeout=SHORT_TIMEOUT,
    )

    sell_order_ids = _extract_sell_ids(sells)

    if sell_order_ids:
        workflow.logger.info(
            f"[{account.upper()}] Waiting for {len(sell_order_ids)} sell orders..."
        )
        deadline = workflow.now() + SELL_DEADLINE

        while workflow.now() < deadline:
            statuses = await workflow.execute_activity(
                check_order_statuses,
                args=[account, sell_order_ids],
                start_to_close_timeout=SHORT_TIMEOUT,
            )
            all_terminal = all(
                s.get("status", "").lower() in TERMINAL_STATUSES for s in statuses
            )

            if all_terminal:
                workflow.logger.info(f"[{account.upper()}] All sell orders terminal.")
                break

            workflow.logger.info(
                f"[{account.upper()}] Sells still pending, sleeping 15 min..."
            )
            await workflow.sleep(SELL_POLL_INTERVAL)
        else:
            workflow.logger.warning(
                f"[{account.upper()}] Sell deadline reached (48h), proceeding to buys."
            )

    buy_resp = _make_buy_response(buy_orders, original_orders)
    buy_submit = await workflow.execute_activity(
        submit_activity,
        args=[buy_resp],
        start_to_close_timeout=SHORT_TIMEOUT,
    )

    return _combine_submit(sell_submit, buy_submit)


def _extract_sell_ids(
    sells: GenerateOrdersResponse | SkippedOrdersResponse,
) -> list[str]:
    """Extract client_order_ids for sell orders from a response."""
    if isinstance(sells, SkippedOrdersResponse) or getattr(sells, "skipped", False):
        return []
    return [o.client_order_id for o in sells.orders if o.side == "sell"]


@workflow.defn
class USWeeklyAllocationWorkflow:
    @workflow.run
    async def run(self) -> dict:
        now_ist = workflow.now().astimezone()
        as_of_date = now_ist.strftime("%Y-%m-%d")
        run_id = f"paper:{as_of_date}"

        attempt = await workflow.execute_activity(
            resolve_next_attempt,
            args=[run_id, as_of_date],
            start_to_close_timeout=SHORT_TIMEOUT,
        )

        workflow.logger.info(
            f"Starting US weekly allocation pipeline (attempt={attempt})..."
        )

        # Phase 0: Get active symbols + portfolios (parallel)
        (
            active_symbols,
            sac_portfolio,
            hrp_portfolio,
        ) = await asyncio.gather(
            workflow.execute_activity(
                get_active_symbols, start_to_close_timeout=SHORT_TIMEOUT
            ),
            workflow.execute_activity(
                get_sac_portfolio, start_to_close_timeout=SHORT_TIMEOUT
            ),
            workflow.execute_activity(
                get_hrp_portfolio, start_to_close_timeout=SHORT_TIMEOUT
            ),
        )

        symbols = active_symbols.symbols
        run_sac = sac_portfolio.open_orders_count == 0
        run_hrp = hrp_portfolio.open_orders_count == 0

        skipped_algorithms = []
        if not run_sac:
            skipped_algorithms.append("SAC")
        if not run_hrp:
            skipped_algorithms.append("HRP")

        # Phase 1: Get signals + forecasts (parallel)
        fundamentals, news, lstm, patchtst = await asyncio.gather(
            workflow.execute_activity(
                get_fundamentals,
                args=[symbols],
                start_to_close_timeout=INFERENCE_TIMEOUT,
                retry_policy=RetryPolicy(maximum_attempts=3),
            ),
            workflow.execute_activity(
                get_news_sentiment,
                args=[symbols, as_of_date, run_id],
                start_to_close_timeout=INFERENCE_TIMEOUT,
                retry_policy=RetryPolicy(maximum_attempts=3),
            ),
            workflow.execute_activity(
                get_lstm_forecast,
                args=[as_of_date, symbols],
                start_to_close_timeout=INFERENCE_TIMEOUT,
                retry_policy=RetryPolicy(maximum_attempts=3),
            ),
            workflow.execute_activity(
                get_patchtst_forecast,
                args=[as_of_date, symbols],
                start_to_close_timeout=INFERENCE_TIMEOUT,
                retry_policy=RetryPolicy(maximum_attempts=3),
            ),
        )

        target_week_start = lstm.target_week_start or as_of_date
        target_week_end = lstm.target_week_end or as_of_date

        # Phase 2: Run allocators (parallel, conditional)
        alloc_futures = []
        if run_sac:
            alloc_futures.append(
                workflow.execute_activity(
                    infer_sac,
                    args=[sac_portfolio, as_of_date],
                    start_to_close_timeout=INFERENCE_TIMEOUT,
                )
            )
        if run_hrp:
            alloc_futures.append(
                workflow.execute_activity(
                    allocate_hrp,
                    args=[symbols, as_of_date],
                    start_to_close_timeout=INFERENCE_TIMEOUT,
                )
            )

        alloc_results = await asyncio.gather(*alloc_futures) if alloc_futures else []

        idx = 0
        sac_alloc = (
            alloc_results[idx] if run_sac else SkippedAllocation(algorithm="sac")
        )
        if run_sac:
            idx += 1
        hrp_alloc = (
            alloc_results[idx] if run_hrp else SkippedAllocation(algorithm="hrp")
        )

        # Phase 3: Generate orders + store experience (parallel)
        sac_orders, hrp_orders = await asyncio.gather(
            workflow.execute_activity(
                generate_orders_sac,
                args=[sac_alloc, sac_portfolio, run_id, attempt],
                start_to_close_timeout=SHORT_TIMEOUT,
            ),
            workflow.execute_activity(
                generate_orders_hrp,
                args=[hrp_alloc, hrp_portfolio, run_id, attempt],
                start_to_close_timeout=SHORT_TIMEOUT,
            ),
        )

        # Store experience (fire and forget for RL algorithms)
        experience_futures = []
        if run_sac:
            experience_futures.append(
                workflow.execute_activity(
                    store_experience_sac,
                    args=[
                        run_id,
                        target_week_start,
                        target_week_end,
                        sac_alloc,
                        sac_portfolio,
                        news,
                        fundamentals,
                        lstm,
                        patchtst,
                    ],
                    start_to_close_timeout=SHORT_TIMEOUT,
                )
            )
        if experience_futures:
            await asyncio.gather(*experience_futures)

        # Phase 4: Per-algorithm sell-wait-buy (parallel per account)
        sac_sells, sac_buys = _split_orders_by_side(sac_orders)
        hrp_sells, hrp_buys = _split_orders_by_side(hrp_orders)

        sac_submit, hrp_submit = await asyncio.gather(
            _sell_wait_buy("sac", sac_sells, sac_buys, sac_orders, submit_orders_sac),
            _sell_wait_buy("hrp", hrp_sells, hrp_buys, hrp_orders, submit_orders_hrp),
        )

        # Phase 5: Get order history + update execution
        if run_sac:
            sac_history = await workflow.execute_activity(
                get_order_history_sac,
                args=[target_week_start],
                start_to_close_timeout=SHORT_TIMEOUT,
            )
            await workflow.execute_activity(
                update_execution_sac,
                args=[run_id, sac_orders, sac_history],
                start_to_close_timeout=SHORT_TIMEOUT,
            )

        # Phase 6: Generate summary + send email
        summary = await workflow.execute_activity(
            generate_summary,
            args=[lstm, patchtst, news, fundamentals, hrp_alloc, sac_alloc],
            start_to_close_timeout=SHORT_TIMEOUT,
        )

        email_result = await workflow.execute_activity(
            send_weekly_email,
            args=[
                summary,
                lstm,
                patchtst,
                hrp_alloc,
                sac_alloc,
                sac_submit,
                hrp_submit,
                target_week_start,
                target_week_end,
                as_of_date,
                skipped_algorithms,
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
            "hrp": {
                "orders_submitted": getattr(hrp_submit, "orders_submitted", 0),
                "skipped": not run_hrp,
            },
            "email": {
                "is_success": email_result.is_success,
                "subject": email_result.subject,
            },
        }


def _combine_submit(sell_submit, buy_submit):
    """Combine sell + buy submit results into a single response for email."""
    if isinstance(sell_submit, SkippedSubmitResponse):
        return buy_submit
    if isinstance(buy_submit, SkippedSubmitResponse):
        return sell_submit
    from models import SubmitOrdersResponse

    return SubmitOrdersResponse(
        account=sell_submit.account,
        orders_submitted=sell_submit.orders_submitted + buy_submit.orders_submitted,
        orders_failed=sell_submit.orders_failed + buy_submit.orders_failed,
        skipped=False,
        results=list(sell_submit.results) + list(buy_submit.results),
    )

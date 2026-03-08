"""US weekly allocation workflow with durable sell-wait-buy.

Runs every Monday at 18:00 IST (11:00 UTC). Single workflow replaces the
3-flow Prefect hack (sells flow + monitor cron + buys flow).

Phases:
0. Get active symbols + Alpaca portfolios (parallel)
1. Get signals + forecasts (parallel)
2. Run allocators: PPO, SAC, HRP (parallel, conditional on open orders)
3. Generate orders + store experience
4. Submit sell orders
5. Poll with durable sleep until sells are terminal (or 48h timeout)
6. Submit buy orders
7. Get order history + update execution
8. Generate LLM summary + send email
"""

import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from activities.execution import (
        generate_orders_hrp,
        generate_orders_ppo,
        generate_orders_sac,
        store_experience_ppo,
        store_experience_sac,
        update_execution_ppo,
        update_execution_sac,
    )
    from activities.inference import (
        allocate_hrp,
        get_fundamentals,
        get_lstm_forecast,
        get_news_sentiment,
        get_patchtst_forecast,
        infer_ppo,
        infer_sac,
    )
    from activities.portfolio import (
        check_order_statuses,
        get_active_symbols,
        get_hrp_portfolio,
        get_order_history_ppo,
        get_order_history_sac,
        get_ppo_portfolio,
        get_sac_portfolio,
        resolve_next_attempt,
        submit_orders_hrp,
        submit_orders_ppo,
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
INFERENCE_TIMEOUT = timedelta(minutes=10)
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
            ppo_portfolio,
            sac_portfolio,
            hrp_portfolio,
        ) = await asyncio.gather(
            workflow.execute_activity(
                get_active_symbols, start_to_close_timeout=SHORT_TIMEOUT
            ),
            workflow.execute_activity(
                get_ppo_portfolio, start_to_close_timeout=SHORT_TIMEOUT
            ),
            workflow.execute_activity(
                get_sac_portfolio, start_to_close_timeout=SHORT_TIMEOUT
            ),
            workflow.execute_activity(
                get_hrp_portfolio, start_to_close_timeout=SHORT_TIMEOUT
            ),
        )

        symbols = active_symbols.symbols
        run_ppo = ppo_portfolio.open_orders_count == 0
        run_sac = sac_portfolio.open_orders_count == 0
        run_hrp = hrp_portfolio.open_orders_count == 0

        skipped_algorithms = []
        if not run_ppo:
            skipped_algorithms.append("PPO")
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
                args=[as_of_date],
                start_to_close_timeout=INFERENCE_TIMEOUT,
                retry_policy=RetryPolicy(maximum_attempts=3),
            ),
            workflow.execute_activity(
                get_patchtst_forecast,
                args=[as_of_date],
                start_to_close_timeout=INFERENCE_TIMEOUT,
                retry_policy=RetryPolicy(maximum_attempts=3),
            ),
        )

        target_week_start = lstm.target_week_start or as_of_date
        target_week_end = lstm.target_week_end or as_of_date

        # Phase 2: Run allocators (parallel, conditional)
        alloc_futures = []
        if run_ppo:
            alloc_futures.append(
                workflow.execute_activity(
                    infer_ppo,
                    args=[ppo_portfolio, as_of_date],
                    start_to_close_timeout=INFERENCE_TIMEOUT,
                )
            )
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
                    args=[as_of_date],
                    start_to_close_timeout=INFERENCE_TIMEOUT,
                )
            )

        alloc_results = await asyncio.gather(*alloc_futures) if alloc_futures else []

        idx = 0
        ppo_alloc = (
            alloc_results[idx] if run_ppo else SkippedAllocation(algorithm="ppo")
        )
        if run_ppo:
            idx += 1
        sac_alloc = (
            alloc_results[idx] if run_sac else SkippedAllocation(algorithm="sac")
        )
        if run_sac:
            idx += 1
        hrp_alloc = (
            alloc_results[idx] if run_hrp else SkippedAllocation(algorithm="hrp")
        )

        # Phase 3: Generate orders + store experience (parallel)
        ppo_orders, sac_orders, hrp_orders = await asyncio.gather(
            workflow.execute_activity(
                generate_orders_ppo,
                args=[ppo_alloc, ppo_portfolio, run_id, attempt],
                start_to_close_timeout=SHORT_TIMEOUT,
            ),
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
        if run_ppo:
            experience_futures.append(
                workflow.execute_activity(
                    store_experience_ppo,
                    args=[
                        run_id,
                        target_week_start,
                        target_week_end,
                        ppo_alloc,
                        ppo_portfolio,
                        news,
                        fundamentals,
                        lstm,
                        patchtst,
                    ],
                    start_to_close_timeout=SHORT_TIMEOUT,
                )
            )
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

        # Phase 4: Split into sells and buys
        ppo_sells, ppo_buys = _split_orders_by_side(ppo_orders)
        sac_sells, sac_buys = _split_orders_by_side(sac_orders)
        hrp_sells, hrp_buys = _split_orders_by_side(hrp_orders)

        # Submit sells
        ppo_sell_submit, sac_sell_submit, hrp_sell_submit = await asyncio.gather(
            workflow.execute_activity(
                submit_orders_ppo,
                args=[ppo_sells],
                start_to_close_timeout=SHORT_TIMEOUT,
            ),
            workflow.execute_activity(
                submit_orders_sac,
                args=[sac_sells],
                start_to_close_timeout=SHORT_TIMEOUT,
            ),
            workflow.execute_activity(
                submit_orders_hrp,
                args=[hrp_sells],
                start_to_close_timeout=SHORT_TIMEOUT,
            ),
        )

        # Phase 5: Poll with durable sleep until sells are terminal
        all_sell_order_ids = _collect_sell_order_ids(ppo_sells, sac_sells, hrp_sells)

        if all_sell_order_ids:
            workflow.logger.info(
                f"Waiting for {len(all_sell_order_ids)} sell orders to fill..."
            )
            deadline = workflow.now() + SELL_DEADLINE

            while workflow.now() < deadline:
                all_terminal = True
                for account, order_ids in all_sell_order_ids.items():
                    if not order_ids:
                        continue
                    statuses = await workflow.execute_activity(
                        check_order_statuses,
                        args=[account, order_ids],
                        start_to_close_timeout=SHORT_TIMEOUT,
                    )
                    for s in statuses:
                        if s.get("status", "").lower() not in TERMINAL_STATUSES:
                            all_terminal = False
                            break
                    if not all_terminal:
                        break

                if all_terminal:
                    workflow.logger.info("All sell orders are terminal.")
                    break

                workflow.logger.info("Sell orders still pending, sleeping 15 min...")
                await workflow.sleep(SELL_POLL_INTERVAL)
            else:
                workflow.logger.warning(
                    "Sell deadline reached (48h), proceeding to buys."
                )

        # Phase 6: Submit buys
        ppo_buy_resp = _make_buy_response(ppo_buys, ppo_orders)
        sac_buy_resp = _make_buy_response(sac_buys, sac_orders)
        hrp_buy_resp = _make_buy_response(hrp_buys, hrp_orders)

        ppo_buy_submit, sac_buy_submit, hrp_buy_submit = await asyncio.gather(
            workflow.execute_activity(
                submit_orders_ppo,
                args=[ppo_buy_resp],
                start_to_close_timeout=SHORT_TIMEOUT,
            ),
            workflow.execute_activity(
                submit_orders_sac,
                args=[sac_buy_resp],
                start_to_close_timeout=SHORT_TIMEOUT,
            ),
            workflow.execute_activity(
                submit_orders_hrp,
                args=[hrp_buy_resp],
                start_to_close_timeout=SHORT_TIMEOUT,
            ),
        )

        # Combine sell + buy submit results for email
        ppo_submit = _combine_submit(ppo_sell_submit, ppo_buy_submit)
        sac_submit = _combine_submit(sac_sell_submit, sac_buy_submit)
        hrp_submit = _combine_submit(hrp_sell_submit, hrp_buy_submit)

        # Phase 7: Get order history + update execution
        if run_ppo:
            ppo_history = await workflow.execute_activity(
                get_order_history_ppo,
                args=[target_week_start],
                start_to_close_timeout=SHORT_TIMEOUT,
            )
            await workflow.execute_activity(
                update_execution_ppo,
                args=[run_id, ppo_orders, ppo_history],
                start_to_close_timeout=SHORT_TIMEOUT,
            )
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

        # Phase 8: Generate summary + send email
        summary = await workflow.execute_activity(
            generate_summary,
            args=[lstm, patchtst, news, fundamentals, hrp_alloc, sac_alloc, ppo_alloc],
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
                ppo_alloc,
                ppo_submit,
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
            "ppo": {
                "orders_submitted": getattr(ppo_submit, "orders_submitted", 0),
                "skipped": not run_ppo,
            },
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


def _collect_sell_order_ids(ppo_sells, sac_sells, hrp_sells) -> dict[str, list[str]]:
    """Collect all sell order client_order_ids grouped by account."""
    result: dict[str, list[str]] = {}
    for account, sells in [("ppo", ppo_sells), ("sac", sac_sells), ("hrp", hrp_sells)]:
        if isinstance(sells, SkippedOrdersResponse) or getattr(sells, "skipped", False):
            continue
        ids = [o.client_order_id for o in sells.orders if o.side == "sell"]
        if ids:
            result[account] = ids
    return result


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

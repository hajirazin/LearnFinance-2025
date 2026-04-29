"""Shared sell-wait-buy machinery for Alpaca-backed Temporal workflows.

Extracted from ``us_weekly_allocation`` so the new ``us_double_hrp``
workflow (and any future US strategy) can reuse the same durable
"submit sells -> poll until terminal -> submit buys" cycle without
duplicating logic.

Key invariants preserved during extraction:
- ``workflow.now()``/``workflow.sleep()`` semantics so durable timers
  survive worker restarts.
- ``workflow.execute_activity`` with ``SHORT_TIMEOUT`` for every
  external call.
- The 48h sell-deadline fallback that proceeds to buys even if some
  sells never reach terminal.
- ``SkippedOrdersResponse``/``SkippedSubmitResponse`` short-circuits.
"""

from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from activities.portfolio import check_order_statuses
    from models import (
        GenerateOrdersResponse,
        OrderModel,
        SkippedOrdersResponse,
        SkippedSubmitResponse,
    )

# Reused constants (single source of truth).
SHORT_TIMEOUT = timedelta(minutes=5)
SELL_POLL_INTERVAL = timedelta(minutes=15)
SELL_DEADLINE = timedelta(hours=48)

TERMINAL_STATUSES = {"filled", "canceled", "expired", "rejected", "replaced"}


def split_orders_by_side(
    orders_resp: GenerateOrdersResponse | SkippedOrdersResponse,
) -> tuple[GenerateOrdersResponse | SkippedOrdersResponse, list[OrderModel]]:
    """Split orders into a sell-only response and a buy order list.

    Returns ``(sell_only_response, buy_orders_list)``. If the input is a
    skipped response, returns it as-is plus an empty buy list.
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


def make_buy_response(
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


def extract_sell_ids(
    sells: GenerateOrdersResponse | SkippedOrdersResponse,
) -> list[str]:
    """Extract client_order_ids for sell orders from a response."""
    if isinstance(sells, SkippedOrdersResponse) or getattr(sells, "skipped", False):
        return []
    return [o.client_order_id for o in sells.orders if o.side == "sell"]


def combine_submit(sell_submit, buy_submit):
    """Combine sell + buy submit results into a single response for email."""
    if isinstance(sell_submit, SkippedSubmitResponse):
        return buy_submit
    if isinstance(buy_submit, SkippedSubmitResponse):
        return sell_submit
    with workflow.unsafe.imports_passed_through():
        from models import SubmitOrdersResponse

    return SubmitOrdersResponse(
        account=sell_submit.account,
        orders_submitted=sell_submit.orders_submitted + buy_submit.orders_submitted,
        orders_failed=sell_submit.orders_failed + buy_submit.orders_failed,
        skipped=False,
        results=list(sell_submit.results) + list(buy_submit.results),
    )


async def sell_wait_buy(
    account: str,
    sells: GenerateOrdersResponse | SkippedOrdersResponse,
    buy_orders: list[OrderModel],
    original_orders: GenerateOrdersResponse | SkippedOrdersResponse,
    submit_activity,
):
    """Run the full sell -> poll -> buy cycle for a single Alpaca account.

    Each algorithm has its own Alpaca account, so multiple sell-wait-buy
    pipelines can run in parallel via ``asyncio.gather``.
    """
    sell_submit = await workflow.execute_activity(
        submit_activity,
        args=[sells],
        start_to_close_timeout=SHORT_TIMEOUT,
    )

    sell_order_ids = extract_sell_ids(sells)

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

    buy_resp = make_buy_response(buy_orders, original_orders)
    buy_submit = await workflow.execute_activity(
        submit_activity,
        args=[buy_resp],
        start_to_close_timeout=SHORT_TIMEOUT,
    )

    return combine_submit(sell_submit, buy_submit)

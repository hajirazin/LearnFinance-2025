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

Polling cadence (replaces the legacy flat 15-min loop):
- After the sells are submitted, the helper fetches the Alpaca market
  clock once. If the market is currently closed AND ``next_open`` is
  still in the future, the workflow sleeps until exactly ``next_open``
  (one durable big sleep, not a tight loop).
- Once the market is open, the helper polls every ``POLL_INTERVAL``
  (1 min) until all sells reach a terminal status or the 48h
  ``SELL_DEADLINE`` is hit. The deadline loop's else-branch (proceed
  to buys on timeout) is preserved.
"""

from datetime import datetime, timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from activities.portfolio import check_order_statuses, get_alpaca_clock
    from models import (
        GenerateOrdersResponse,
        OrderModel,
        SkippedOrdersResponse,
        SkippedSubmitResponse,
    )

# Reused constants (single source of truth).
SHORT_TIMEOUT = timedelta(minutes=5)
POLL_INTERVAL = timedelta(minutes=1)
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
    check_status_activity=check_order_statuses,
):
    """Run the full sell -> poll -> buy cycle for a single broker account.

    Each algorithm has its own broker account, so multiple
    sell-wait-buy pipelines can run in parallel via ``asyncio.gather``.

    ``submit_activity`` is the broker-specific submit activity (e.g.
    ``submit_orders_sac`` for Alpaca, ``submit_orders_ibkr_sac_halal``
    for IBKR). ``check_status_activity`` is its sibling for status
    polling -- defaults to the Alpaca-backed
    :func:`activities.portfolio.check_order_statuses` to keep existing
    callers working unchanged; the IBKR-routed
    ``USSACHalalAllocationWorkflow`` passes ``check_order_statuses_ibkr``
    explicitly so the helper never branches on ``account``.
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

        # One-shot clock check: sleep until exactly the next NYSE open
        # if the market is currently closed. Mathematically this is
        # ``max(0, next_open - now)`` -- a strict equality with the
        # advertised open time, no lead-time fudge -- so the first
        # status poll fires the moment the market opens.
        clock = await workflow.execute_activity(
            get_alpaca_clock,
            start_to_close_timeout=SHORT_TIMEOUT,
        )
        if not clock.is_open:
            next_open = datetime.fromisoformat(clock.next_open)
            wait = next_open - workflow.now()
            if wait > timedelta(0):
                workflow.logger.info(
                    f"[{account.upper()}] Sleeping {wait} until market open "
                    f"({next_open.isoformat()})"
                )
                await workflow.sleep(wait)

        deadline = workflow.now() + SELL_DEADLINE

        while workflow.now() < deadline:
            statuses = await workflow.execute_activity(
                check_status_activity,
                args=[account, sell_order_ids],
                start_to_close_timeout=SHORT_TIMEOUT,
            )
            statuses_by_id = {
                status.get("client_order_id"): status
                for status in statuses
                if status.get("client_order_id")
            }
            all_terminal = all(
                order_id in statuses_by_id
                and statuses_by_id[order_id].get("status", "").lower()
                in TERMINAL_STATUSES
                for order_id in sell_order_ids
            )

            if all_terminal:
                workflow.logger.info(f"[{account.upper()}] All sell orders terminal.")
                break

            workflow.logger.info(
                f"[{account.upper()}] Sells still pending, sleeping {POLL_INTERVAL}..."
            )
            await workflow.sleep(POLL_INTERVAL)
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

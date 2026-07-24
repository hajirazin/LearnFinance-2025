"""Interactive Brokers trading endpoints.

Parallel surface to ``/alpaca/*`` for accounts that route through
IBKR's TWS API (currently only ``sac_halal``). Broker selection is a
**Temporal-side** concern -- the workflow picks the URL prefix per
account; brain_api never branches on "which broker" for an account.

Mirrors the Alpaca route shape 1:1 so the Temporal activities are a
drop-in URL/body swap:

- ``GET  /ibkr/portfolio``       -> :class:`PortfolioResponse`
- ``POST /ibkr/submit-orders``   -> :class:`SubmitOrdersResponse`
- ``GET  /ibkr/order-history``   -> ``list[OrderHistoryItem]``

The Pydantic response shapes are intentionally re-used from
``brain_api.routes.alpaca`` so a single broker-agnostic
``PortfolioResponse`` model serves both routes (the temporal worker
already deserializes one shape into ``AlpacaPortfolioResponse``;
keeping field parity means we don't need a separate IBKR Pydantic
class for the worker either).

Idempotency note
----------------
IBKR will NOT auto-reject duplicate ``Order.orderRef`` values (unlike
Alpaca's per-account ``client_order_id`` dedup). The pre-submit dedup
gate inside :func:`submit_orders` is therefore part of brain_api's
contract, not the broker's. Two checks happen before each placement:

1. The local Postgres-style ledger (``ibkr_submitted_orders`` table
   in ``data/ibkr/submitted_orders.db``) is queried for the
   ``order_ref`` -- a hit means a previous workflow attempt already
   submitted this order_ref, possibly across daily-gateway-restart
   boundaries.
2. The gateway's own open-trades book is scanned for the
   ``order_ref`` -- a hit means the order is currently in flight.

Either hit short-circuits to a ``deduped`` result instead of placing
a duplicate.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from brain_api.core.ibkr_client import (
    IBKROrderSpec,
    IBKRSubmitResult,
    get_connection_config,
    get_order_status,
    get_portfolio,
    get_session_status,
    list_open_order_refs,
    submit_order,
)
from brain_api.routes.alpaca import (
    OrderHistoryItem,
    OrderSubmitResult,
    OrderToSubmit,
    PortfolioResponse,
    PositionResponse,
    SubmitOrdersResponse,
)
from brain_api.storage.ibkr_orders import (
    IBKROrderLedger,
    SubmittedOrderRow,
    get_ibkr_order_ledger,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class IBKRAccount(str, Enum):
    """Supported IBKR trading accounts.

    Only ``sac_halal`` is wired today (cloned from the legacy halal SAC
    workflow). Adding a second account here also requires a matching
    ``IBKR_{ACCOUNT}_*`` env block and (if it should run live) a
    second ``ibkr-gateway-live`` compose service on port 4001.
    """

    SAC_HALAL = "sac_halal"


class SubmitOrdersRequest(BaseModel):
    """Request model for submitting multiple orders to IBKR."""

    account: IBKRAccount = Field(..., description="IBKR account (currently sac_halal)")
    orders: list[OrderToSubmit] = Field(
        default_factory=list, description="Orders to submit"
    )


# ============================================================================
# Endpoints
# ============================================================================


@router.get("/portfolio", response_model=PortfolioResponse)
def get_portfolio_route(
    account: Annotated[
        IBKRAccount, Query(..., description="IBKR account (currently sac_halal)")
    ],
    target_currency: Annotated[
        str, Query(description="Currency to calculate cash equivalent in")
    ] = "USD",
) -> PortfolioResponse:
    """Get cash + positions + open orders count for an IBKR account.

    Routes ``account`` through ``IBKR_{ACCOUNT}_*`` env vars (host,
    port, client id, account code), opens (or reuses) a cached
    connection to the local IB Gateway, and returns the same
    :class:`PortfolioResponse` shape as ``/alpaca/portfolio``.

    Raises:
        HTTPException 500: if any ``IBKR_{ACCOUNT}_*`` env var is
            missing or malformed.
        HTTPException 503: if the IB Gateway connection cannot be
            established.
    """
    logger.info(f"[IBKR] Fetching portfolio for account {account.value}")
    try:
        config = get_connection_config(account.value)
    except ValueError as e:
        logger.error(f"[IBKR] Misconfigured account {account.value}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

    try:
        portfolio = get_portfolio(config, target_currency=target_currency)
    except ConnectionError as e:
        logger.error(f"[IBKR] Gateway connection failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"IB Gateway unreachable for account {account.value}: {e}",
        ) from e
    except Exception as e:
        logger.error(f"[IBKR] Unexpected error fetching portfolio: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=f"Failed to fetch IBKR portfolio: {e!s}",
        ) from e

    return PortfolioResponse(
        cash=portfolio.cash,
        currency=portfolio.currency,
        cash_balances=portfolio.cash_balances,
        positions=[
            PositionResponse(
                symbol=p.symbol,
                qty=p.qty,
                market_value=p.market_value,
            )
            for p in portfolio.positions
        ],
        open_orders_count=portfolio.open_orders_count,
    )


@router.post("/submit-orders", response_model=SubmitOrdersResponse)
def submit_orders(
    request: SubmitOrdersRequest,
    ledger: Annotated[IBKROrderLedger, Depends(get_ibkr_order_ledger)],
) -> SubmitOrdersResponse:
    """Submit orders to IBKR with broker-agnostic dedup guardrail.

    For each order, the dedup gate checks both:

    1. The local ledger (``ibkr_submitted_orders``) -- catches
       previously-submitted ``order_ref`` even after the gateway's
       daily session reset.
    2. The gateway's open-trades book -- catches in-flight orders that
       haven't been recorded yet (e.g. a half-completed previous run).

    A hit in either short-circuits to a ``status='deduped'`` result;
    the order is NOT placed. Per AGENTS.md "Order idempotency" the
    deterministic ``client_order_id`` is the dedup key, mirrored onto
    IBKR's ``Order.orderRef``.

    Sells are submitted before buys to free up cash, mirroring the
    Alpaca route's ordering convention.
    """
    logger.info(
        f"[IBKR] Submitting {len(request.orders)} orders for account {request.account.value}"
    )

    if not request.orders:
        return SubmitOrdersResponse(
            account=request.account.value,
            orders_submitted=0,
            orders_failed=0,
            skipped=False,
            results=[],
        )

    try:
        config = get_connection_config(request.account.value)
    except ValueError as e:
        logger.error(f"[IBKR] Misconfigured account {request.account.value}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

    # Pre-warm + check open trades on the gateway in one connection
    # roundtrip (raises 503 if gateway unreachable -- catch and convert
    # at the boundary for parity with the Alpaca route).
    try:
        if not get_session_status(config):
            raise ConnectionError("Gateway session is not authenticated")
        open_refs = list_open_order_refs(config)
    except ConnectionError as e:
        logger.error(f"[IBKR] Gateway connection failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"IB Gateway unreachable for account {request.account.value}: {e}",
        ) from e

    sorted_orders = sorted(request.orders, key=lambda o: (o.side != "sell", o.symbol))

    results: list[OrderSubmitResult] = []
    orders_submitted = 0
    orders_failed = 0

    for order in sorted_orders:
        # Dedup gate -- ledger first (cheap, includes terminal orders);
        # gateway open-trades scan second (catches the gap between
        # placement and ledger insert on a previously-failed attempt).
        if ledger.has_order_ref(order.client_order_id) or (
            order.client_order_id in open_refs
        ):
            logger.info(
                f"[IBKR] Skipping duplicate client_order_id={order.client_order_id}"
            )
            results.append(
                OrderSubmitResult(
                    id=None,
                    client_order_id=order.client_order_id,
                    symbol=order.symbol,
                    status="deduped",
                    error=None,
                )
            )
            continue

        spec = IBKROrderSpec(
            symbol=order.symbol,
            qty=order.qty,
            side=order.side,
            order_type=order.type,
            time_in_force=order.time_in_force,
            limit_price=order.limit_price,
            client_order_id=order.client_order_id,
            currency=order.currency,
            cash_qty=order.cash_qty,
        )
        try:
            outcome: IBKRSubmitResult = submit_order(config, spec)
        except ValueError as e:
            # Bad order (unsupported side / type / missing limit_price).
            results.append(
                OrderSubmitResult(
                    id=None,
                    client_order_id=order.client_order_id,
                    symbol=order.symbol,
                    status="rejected",
                    error=str(e),
                )
            )
            orders_failed += 1
            logger.warning(
                f"[IBKR] Order rejected at mapping: {order.client_order_id}: {e}"
            )
            continue
        except Exception as e:
            results.append(
                OrderSubmitResult(
                    id=None,
                    client_order_id=order.client_order_id,
                    symbol=order.symbol,
                    status="error",
                    error=str(e),
                )
            )
            orders_failed += 1
            logger.error(
                f"[IBKR] placeOrder failed: {order.client_order_id}: {e}",
                exc_info=True,
            )
            continue

        # Mirror the submission into the local ledger BEFORE acknowledging
        # success so a crash between placeOrder and ledger.record never
        # leaves an orphan placement on the gateway with no local trail.
        run_id, attempt = _parse_run_id_and_attempt(order.client_order_id)
        ledger.record_submission(
            SubmittedOrderRow(
                account=request.account.value,
                run_id=run_id,
                attempt=attempt,
                symbol=order.symbol,
                side=order.side,
                qty=order.qty,
                limit_price=order.limit_price,
                order_ref=order.client_order_id,
                ibkr_perm_id=outcome.perm_id,
                status=outcome.status,
                filled_qty=None,
                filled_avg_price=None,
            )
        )

        if outcome.error is not None:
            results.append(
                OrderSubmitResult(
                    id=str(outcome.perm_id) if outcome.perm_id else None,
                    client_order_id=order.client_order_id,
                    symbol=order.symbol,
                    status=outcome.status,
                    error=outcome.error,
                )
            )
            orders_failed += 1
        else:
            results.append(
                OrderSubmitResult(
                    id=str(outcome.perm_id) if outcome.perm_id else None,
                    client_order_id=order.client_order_id,
                    symbol=order.symbol,
                    status=outcome.status,
                    error=None,
                )
            )
            orders_submitted += 1
            logger.debug(
                f"[IBKR] Order submitted: {order.client_order_id} permId={outcome.perm_id}"
            )

    logger.info(
        f"[IBKR] Order submission complete for {request.account.value}: "
        f"{orders_submitted} submitted, {orders_failed} failed"
    )

    return SubmitOrdersResponse(
        account=request.account.value,
        orders_submitted=orders_submitted,
        orders_failed=orders_failed,
        skipped=False,
        results=results,
    )


@router.get("/order-history", response_model=list[OrderHistoryItem])
def get_order_history(
    ledger: Annotated[IBKROrderLedger, Depends(get_ibkr_order_ledger)],
    account: Annotated[
        IBKRAccount, Query(..., description="IBKR account (currently sac_halal)")
    ],
    after: Annotated[
        str,
        Query(..., description="ISO date to fetch orders submitted on/after"),
    ],
    sync_broker: Annotated[
        bool,
        Query(
            description="If True, poll the live broker API for latest statuses before returning"
        ),
    ] = False,
) -> list[OrderHistoryItem]:
    """Read order history from the local IBKR ledger (NOT the gateway).

    The IB Gateway's own ``reqCompletedOrders()`` only returns the
    current TWS session, which resets daily on the IBC soft-restart
    cadence. ``resolve_next_attempt`` and ``check_order_statuses`` in
    Temporal need order visibility for at least the past trading week,
    so we serve this endpoint from the local ledger that mirrored
    every submission as it happened.

    If `sync_broker` is True, it will reach out to the broker API to
    get the live status for each non-terminal order before returning,
    updating the local ledger in the process.

    Mirrors the Alpaca ``/alpaca/order-history`` shape so the
    workflow's regex on ``client_order_id`` is unchanged.
    """
    logger.info(
        f"[IBKR] Fetching order history for account {account.value} after {after} (sync={sync_broker})"
    )
    rows = ledger.list_after(account.value, after)

    if sync_broker and rows:
        try:
            config = get_connection_config(account.value)
            # Pre-warm connection for polling
            if not get_session_status(config):
                logger.warning(
                    f"[IBKR] Gateway session not authenticated for {account.value}"
                )

            updates = []
            for row in rows:
                if row.ibkr_perm_id and row.status not in (
                    "Filled",
                    "Cancelled",
                    "Inactive",
                    "ApiCancelled",
                ):
                    try:
                        live_status = get_order_status(config, row.ibkr_perm_id)
                        if live_status:
                            updates.append(
                                (
                                    row.order_ref,
                                    live_status.status,
                                    live_status.filled_qty,
                                    live_status.filled_avg_price,
                                )
                            )
                    except Exception as e:
                        logger.error(
                            f"[IBKR] Failed to sync status for {row.ibkr_perm_id}: {e}"
                        )

            if updates:
                ledger.update_status_batch(updates)
                # Re-fetch after updates to get the latest state
                rows = ledger.list_after(account.value, after)
        except Exception as e:
            logger.error(f"[IBKR] Failed to sync broker statuses: {e}")

    return [
        OrderHistoryItem(
            id=str(r.ibkr_perm_id) if r.ibkr_perm_id else r.order_ref,
            client_order_id=r.order_ref,
            symbol=r.symbol,
            side=r.side,
            status=r.status,
            filled_qty=str(r.filled_qty) if r.filled_qty is not None else None,
            filled_avg_price=(
                str(r.filled_avg_price) if r.filled_avg_price is not None else None
            ),
        )
        for r in rows
    ]


# ============================================================================
# Helpers
# ============================================================================


def _parse_run_id_and_attempt(client_order_id: str) -> tuple[str, int]:
    """Parse the AGENTS.md client_order_id format into ``(run_id, attempt)``.

    Format::

        paper:YYYY-MM-DD:attempt-N:SYMBOL:SIDE                # default
        paper:<universe>:YYYY-MM-DD:attempt-N:SYMBOL:SIDE     # variant

    The dedicated-account variant prefix is allowed only when the
    strategy uses a dedicated broker account (currently only
    ``sac_halal``). The parser splits on ``:attempt-`` rather than a
    fixed segment count so both forms work.

    Falls back to ``(client_order_id, 0)`` if the id does not match the
    expected shape -- safer than raising in the middle of an order
    submission loop, since the ledger row is purely advisory and the
    gateway's permId is what matters for execution tracking.
    """
    try:
        prefix, rest = client_order_id.split(":attempt-", 1)
    except ValueError:
        return client_order_id, 0
    attempt_str, _, _ = rest.partition(":")
    try:
        attempt = int(attempt_str)
    except ValueError:
        return prefix, 0
    return prefix, attempt

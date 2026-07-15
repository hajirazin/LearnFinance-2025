"""Portfolio, order submission, order history, and order status activities."""

import logging
import re

from temporalio import activity

from activities.client import get_client
from models import (
    ActiveSymbolsResponse,
    AlpacaPortfolioResponse,
    GenerateOrdersResponse,
    MarketClockResponse,
    OrderHistoryItem,
    SkippedOrdersResponse,
    SkippedSubmitResponse,
    SubmitOrdersResponse,
)

logger = logging.getLogger(__name__)


def _compute_max_attempt(run_id: str, orders: list[dict]) -> int:
    """Return the highest ``attempt-N`` integer found in a list of orders.

    Shared by :func:`resolve_next_attempt` (Alpaca) and
    :func:`resolve_next_attempt_ibkr` (IBKR). Both order-history routes
    return the same ``OrderHistoryItem`` shape, and the
    ``client_order_id`` format is broker-agnostic (per AGENTS.md
    "Order idempotency"), so the regex aggregation is identical math:

    * default form  ``paper:YYYY-MM-DD:attempt-N:SYMBOL:SIDE``
    * variant form  ``paper:<universe>:YYYY-MM-DD:attempt-N:SYMBOL:SIDE``

    Returns ``0`` if no order matches the prefix (cold start).
    """
    pattern = re.compile(rf"^{re.escape(run_id)}:attempt-(\d+):")
    max_attempt = 0
    for order in orders:
        coid = order.get("client_order_id", "")
        match = pattern.match(coid)
        if match:
            max_attempt = max(max_attempt, int(match.group(1)))
    return max_attempt


@activity.defn
def resolve_next_attempt(
    run_id: str,
    as_of_date: str,
    accounts: list[str] | None = None,
) -> int:
    """Find the max attempt already used in Alpaca orders, return max + 1.

    Parses client_order_id (format: paper:YYYY-MM-DD:attempt-N:SYMBOL:SIDE)
    across the given accounts to avoid duplicate IDs on reruns.

    Each US strategy uses its own Alpaca account, so a workflow should
    only scan its own account(s). Defaults to the SAC+HRP pair (the
    legacy ``USWeeklyAllocationWorkflow`` accounts) so existing callers
    keep working without changes.

    The IBKR equivalent is :func:`resolve_next_attempt_ibkr`. The
    workflow picks the right sibling per broker, mirroring the existing
    ``check_order_statuses`` / ``check_order_statuses_ibkr`` split --
    we deliberately do NOT branch on ``account`` inside the activity
    (per AGENTS.md rule #1, no silent broker-routing fallback).
    """
    if accounts is None:
        accounts = ["sac", "hrp"]

    max_attempt = 0
    for account in accounts:
        with get_client() as client:
            response = client.get(
                "/alpaca/order-history",
                params={"account": account, "after": as_of_date},
            )
            response.raise_for_status()
        max_attempt = max(max_attempt, _compute_max_attempt(run_id, response.json()))

    next_attempt = max_attempt + 1
    logger.info(
        f"Resolved next attempt for {run_id} accounts={accounts}: {next_attempt} "
        f"(max existing: {max_attempt})"
    )
    return next_attempt


@activity.defn
def resolve_next_attempt_ibkr(
    run_id: str,
    as_of_date: str,
    accounts: list[str],
) -> int:
    """IBKR sibling of :func:`resolve_next_attempt`.

    Hits brain_api's ``/ibkr/order-history`` (backed by the local IBKR
    ledger; see ``brain_api/storage/ibkr_orders.py``) instead of the
    Alpaca route. The IB Gateway's own ``reqCompletedOrders()`` only
    covers the current TWS session, so the ledger is the only source
    of truth that survives the daily IBC soft-restart -- which is why
    the IBKR docstring on ``/ibkr/order-history`` explicitly names
    this activity as a consumer.

    Same ``client_order_id`` regex as the Alpaca path (the IBKR ledger
    preserves ``order_ref`` verbatim, so the variant form
    ``paper:<universe>:YYYY-MM-DD:attempt-N:SYMBOL:SIDE`` parses
    identically). ``accounts`` is required (no default) to keep the
    workflow explicit about which IBKR account it owns.
    """
    max_attempt = 0
    for account in accounts:
        with get_client() as client:
            response = client.get(
                "/ibkr/order-history",
                params={"account": account, "after": as_of_date},
            )
            response.raise_for_status()
        max_attempt = max(max_attempt, _compute_max_attempt(run_id, response.json()))

    next_attempt = max_attempt + 1
    logger.info(
        f"Resolved next attempt for {run_id} (IBKR) accounts={accounts}: "
        f"{next_attempt} (max existing: {max_attempt})"
    )
    return next_attempt


@activity.defn
def get_active_symbols(universe: str) -> ActiveSymbolsResponse:
    """Fetch the active symbols from the requested SAC bucket.

    The ``universe`` arg is mandatory (no default) so the two parallel
    A/B SAC workflows -- ``USWeeklyAllocationWorkflow``
    (``halal_filtered``) and ``USSACHalalAllocationWorkflow``
    (``halal``) -- cannot accidentally read from each other's bucket.
    Per AGENTS.md rule #1 (no silent fallbacks).
    """
    logger.info(f"Fetching active symbols from SAC bucket (universe={universe})...")
    with get_client() as client:
        response = client.get("/models/active-symbols", params={"universe": universe})
        response.raise_for_status()
    result = ActiveSymbolsResponse(**response.json())
    logger.info(
        f"Got {len(result.symbols)} active symbols "
        f"(source={result.source_model}, version={result.model_version})"
    )
    return result


@activity.defn
def get_sac_portfolio() -> AlpacaPortfolioResponse:
    """Fetch SAC Alpaca account portfolio (halal_filtered universe)."""
    logger.info("Fetching SAC portfolio from Alpaca...")
    with get_client() as client:
        response = client.get("/alpaca/portfolio", params={"account": "sac"})
        response.raise_for_status()
    result = AlpacaPortfolioResponse(**response.json())
    logger.info(
        f"SAC portfolio: cash=${result.cash:.2f}, "
        f"{len(result.positions)} positions, "
        f"{result.open_orders_count} open orders"
    )
    return result


@activity.defn
def get_ibkr_sac_halal_portfolio() -> AlpacaPortfolioResponse:
    """Fetch SAC halal IBKR account portfolio (legacy halal universe).

    Sibling of :func:`get_sac_portfolio` but routes through the IBKR
    ``/ibkr/*`` surface instead of ``/alpaca/*``. The ``sac_halal``
    workflow trades on a dedicated IBKR paper account (env
    ``IBKR_SAC_HALAL_*``) so positions, cash, and open-order state
    stay disjoint from every Alpaca-backed strategy.

    Reuses the broker-agnostic ``AlpacaPortfolioResponse`` Pydantic
    model because brain_api's ``/ibkr/portfolio`` route returns the
    same shape on purpose (cash, positions, open_orders_count). The
    name ``AlpacaPortfolioResponse`` is a historical artifact -- treat
    it as ``PortfolioResponse``.
    """
    logger.info("Fetching SAC halal portfolio from IBKR...")
    with get_client() as client:
        response = client.get("/ibkr/portfolio", params={"account": "sac_halal"})
        response.raise_for_status()
    result = AlpacaPortfolioResponse(**response.json())
    logger.info(
        f"SAC halal IBKR portfolio: cash=${result.cash:.2f}, "
        f"{len(result.positions)} positions, "
        f"{result.open_orders_count} open orders"
    )
    return result


@activity.defn
def get_hrp_portfolio() -> AlpacaPortfolioResponse:
    """Fetch HRP Alpaca account portfolio."""
    logger.info("Fetching HRP portfolio from Alpaca...")
    with get_client() as client:
        response = client.get("/alpaca/portfolio", params={"account": "hrp"})
        response.raise_for_status()
    result = AlpacaPortfolioResponse(**response.json())
    logger.info(
        f"HRP portfolio: cash=${result.cash:.2f}, "
        f"{len(result.positions)} positions, "
        f"{result.open_orders_count} open orders"
    )
    return result


@activity.defn
def get_dhrp_portfolio() -> AlpacaPortfolioResponse:
    """Fetch Double HRP Alpaca account portfolio (halal_new universe)."""
    logger.info("Fetching DHRP portfolio from Alpaca...")
    with get_client() as client:
        response = client.get("/alpaca/portfolio", params={"account": "dhrp"})
        response.raise_for_status()
    result = AlpacaPortfolioResponse(**response.json())
    logger.info(
        f"DHRP portfolio: cash=${result.cash:.2f}, "
        f"{len(result.positions)} positions, "
        f"{result.open_orders_count} open orders"
    )
    return result


def _submit_orders(
    account: str,
    orders: GenerateOrdersResponse | SkippedOrdersResponse,
) -> SubmitOrdersResponse | SkippedSubmitResponse:
    """Submit orders for the given account. Shared logic for SAC/HRP."""
    if isinstance(orders, SkippedOrdersResponse) or getattr(orders, "skipped", False):
        logger.info(f"{account.upper()} orders skipped")
        return SkippedSubmitResponse(account=account, skipped=True)

    if not orders.orders:
        logger.info(f"No {account.upper()} orders to submit")
        return SubmitOrdersResponse(
            account=account,
            orders_submitted=0,
            orders_failed=0,
            skipped=False,
            results=[],
        )

    logger.info(f"Submitting {len(orders.orders)} {account.upper()} orders...")
    with get_client() as client:
        response = client.post(
            "/alpaca/submit-orders",
            json={
                "account": account,
                "orders": [o.model_dump() for o in orders.orders],
            },
        )
        response.raise_for_status()
    result = SubmitOrdersResponse(**response.json())
    logger.info(
        f"{account.upper()} orders: {result.orders_submitted} submitted, "
        f"{result.orders_failed} failed"
    )
    return result


@activity.defn
def submit_orders_sac(
    orders: GenerateOrdersResponse | SkippedOrdersResponse,
) -> SubmitOrdersResponse | SkippedSubmitResponse:
    """Submit SAC orders to Alpaca (halal_filtered universe)."""
    return _submit_orders("sac", orders)


def _submit_orders_ibkr(
    account: str,
    orders: GenerateOrdersResponse | SkippedOrdersResponse,
) -> SubmitOrdersResponse | SkippedSubmitResponse:
    """Submit orders to brain_api's ``/ibkr/submit-orders`` route.

    Sibling of :func:`_submit_orders` for IBKR-routed accounts. Same
    skip / empty-orders short-circuits, same response shape -- the
    only difference is the URL prefix and the broker that ultimately
    executes the order.
    """
    if isinstance(orders, SkippedOrdersResponse) or getattr(orders, "skipped", False):
        logger.info(f"{account.upper()} (IBKR) orders skipped")
        return SkippedSubmitResponse(account=account, skipped=True)

    if not orders.orders:
        logger.info(f"No {account.upper()} (IBKR) orders to submit")
        return SubmitOrdersResponse(
            account=account,
            orders_submitted=0,
            orders_failed=0,
            skipped=False,
            results=[],
        )

    logger.info(f"Submitting {len(orders.orders)} {account.upper()} (IBKR) orders...")
    with get_client() as client:
        response = client.post(
            "/ibkr/submit-orders",
            json={
                "account": account,
                "orders": [o.model_dump() for o in orders.orders],
            },
        )
        response.raise_for_status()
    result = SubmitOrdersResponse(**response.json())
    logger.info(
        f"{account.upper()} (IBKR) orders: {result.orders_submitted} submitted, "
        f"{result.orders_failed} failed"
    )
    return result


@activity.defn
def submit_orders_ibkr_sac_halal(
    orders: GenerateOrdersResponse | SkippedOrdersResponse,
) -> SubmitOrdersResponse | SkippedSubmitResponse:
    """Submit SAC halal orders to IBKR (legacy halal universe).

    The ``USSACHalalAllocationWorkflow`` passes this activity into
    :func:`workflows._order_execution.sell_wait_buy` as the submitter,
    so sells fire first then buys -- same durable polling cycle as the
    Alpaca-backed workflows.
    """
    return _submit_orders_ibkr("sac_halal", orders)


@activity.defn
def submit_orders_hrp(
    orders: GenerateOrdersResponse | SkippedOrdersResponse,
) -> SubmitOrdersResponse | SkippedSubmitResponse:
    """Submit HRP orders to Alpaca."""
    return _submit_orders("hrp", orders)


@activity.defn
def submit_orders_dhrp(
    orders: GenerateOrdersResponse | SkippedOrdersResponse,
) -> SubmitOrdersResponse | SkippedSubmitResponse:
    """Submit Double HRP orders to Alpaca (dhrp account)."""
    return _submit_orders("dhrp", orders)


@activity.defn
def get_order_history_sac(after_date: str) -> list[OrderHistoryItem]:
    """Fetch SAC order history from Alpaca (halal_filtered universe)."""
    logger.info(f"Fetching SAC order history after {after_date}...")
    with get_client() as client:
        response = client.get(
            "/alpaca/order-history", params={"account": "sac", "after": after_date}
        )
        response.raise_for_status()
    result = [OrderHistoryItem(**o) for o in response.json()]
    logger.info(f"Got {len(result)} SAC orders from history")
    return result


@activity.defn
def get_order_history_ibkr_sac_halal(after_date: str) -> list[OrderHistoryItem]:
    """Fetch SAC halal order history from IBKR (legacy halal universe).

    Reads brain_api's local IBKR order ledger (the IB Gateway's own
    completed-orders feed only covers the current daily session).
    Same response shape as the Alpaca order-history activities so
    downstream consumers (``check_order_statuses_ibkr``,
    ``update_execution_sac``) treat the two interchangeably.
    """
    logger.info(f"Fetching SAC halal IBKR order history after {after_date}...")
    with get_client() as client:
        response = client.get(
            "/ibkr/order-history",
            params={"account": "sac_halal", "after": after_date},
        )
        response.raise_for_status()
    result = [OrderHistoryItem(**o) for o in response.json()]
    logger.info(f"Got {len(result)} SAC halal IBKR orders from history")
    return result


@activity.defn
def get_alpaca_clock() -> MarketClockResponse:
    """Fetch the current Alpaca market clock.

    Wraps brain_api's ``GET /alpaca/clock`` endpoint, which authenticates
    with the generic ``ALPACA_API_KEY`` / ``ALPACA_API_SECRET`` env pair
    (not per-account trading creds -- the clock is account-agnostic
    market data). Used by :func:`workflows._order_execution.sell_wait_buy`
    to sleep until the next NYSE open before polling sell-order status
    at a 1-min cadence.
    """
    logger.info("Fetching Alpaca market clock...")
    with get_client() as client:
        response = client.get("/alpaca/clock")
        response.raise_for_status()
    result = MarketClockResponse(**response.json())
    logger.info(
        f"Alpaca clock: is_open={result.is_open}, "
        f"next_open={result.next_open}, next_close={result.next_close}"
    )
    return result


@activity.defn
def check_order_statuses(account: str, client_order_ids: list[str]) -> list[dict]:
    """Check order statuses using /alpaca/order-history endpoint.

    Fetches recent order history and filters to the given client_order_ids.
    Returns list of {client_order_id, status, filled_qty, filled_avg_price}.

    Used by Alpaca-backed sell-wait-buy loops. The IBKR equivalent is
    :func:`check_order_statuses_ibkr`; ``sell_wait_buy`` takes the
    status-checker as a parameter so the helper stays broker-agnostic
    rather than branching on ``account``.
    """
    logger.info(
        f"Checking {len(client_order_ids)} Alpaca order statuses "
        f"for {account.upper()}..."
    )
    today = activity.info().current_attempt_scheduled_time.strftime("%Y-%m-%d")
    with get_client() as client:
        response = client.get(
            "/alpaca/order-history",
            params={"account": account, "after": today},
        )
        response.raise_for_status()
    all_orders = response.json()
    id_set = set(client_order_ids)
    matched = [o for o in all_orders if o.get("client_order_id") in id_set]
    logger.info(
        f"Got {len(matched)}/{len(client_order_ids)} Alpaca order statuses "
        f"for {account.upper()}"
    )
    return matched


@activity.defn
def check_order_statuses_ibkr(account: str, client_order_ids: list[str]) -> list[dict]:
    """IBKR sibling of :func:`check_order_statuses`.

    Hits brain_api's ``/ibkr/order-history`` (backed by the local
    ledger; see ``brain_api/storage/ibkr_orders.py``) instead of the
    Alpaca route. The shape returned matches Alpaca's so
    ``sell_wait_buy`` can stay broker-agnostic -- it just calls
    whichever status-check activity the workflow handed it.
    """
    logger.info(
        f"Checking {len(client_order_ids)} IBKR order statuses for {account.upper()}..."
    )
    today = activity.info().current_attempt_scheduled_time.strftime("%Y-%m-%d")
    with get_client() as client:
        response = client.get(
            "/ibkr/order-history",
            params={"account": account, "after": today, "sync_broker": "true"},
        )
        response.raise_for_status()
    all_orders = response.json()
    id_set = set(client_order_ids)
    matched = [o for o in all_orders if o.get("client_order_id") in id_set]
    logger.info(
        f"Got {len(matched)}/{len(client_order_ids)} IBKR order statuses "
        f"for {account.upper()}"
    )
    return matched

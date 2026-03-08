"""Portfolio, order submission, order history, and order status activities."""

import logging
import re

from temporalio import activity

from activities.client import get_client
from models import (
    ActiveSymbolsResponse,
    AlpacaPortfolioResponse,
    GenerateOrdersResponse,
    OrderHistoryItem,
    SkippedOrdersResponse,
    SkippedSubmitResponse,
    SubmitOrdersResponse,
)

logger = logging.getLogger(__name__)


@activity.defn
def resolve_next_attempt(run_id: str, as_of_date: str) -> int:
    """Find the max attempt already used in Alpaca orders, return max + 1.

    Parses client_order_id (format: paper:YYYY-MM-DD:attempt-N:SYMBOL:SIDE)
    across all three accounts to avoid duplicate IDs on reruns.
    """
    max_attempt = 0
    pattern = re.compile(rf"^{re.escape(run_id)}:attempt-(\d+):")

    for account in ("ppo", "sac", "hrp"):
        with get_client() as client:
            response = client.get(
                "/alpaca/order-history",
                params={"account": account, "after": as_of_date},
            )
            response.raise_for_status()

        for order in response.json():
            coid = order.get("client_order_id", "")
            match = pattern.match(coid)
            if match:
                max_attempt = max(max_attempt, int(match.group(1)))

    next_attempt = max_attempt + 1
    logger.info(
        f"Resolved next attempt for {run_id}: {next_attempt} "
        f"(max existing: {max_attempt})"
    )
    return next_attempt


@activity.defn
def get_active_symbols() -> ActiveSymbolsResponse:
    """Fetch the active symbols from the current SAC model via brain_api."""
    logger.info("Fetching active symbols from SAC model...")
    with get_client() as client:
        response = client.get("/models/active-symbols")
        response.raise_for_status()
    result = ActiveSymbolsResponse(**response.json())
    logger.info(
        f"Got {len(result.symbols)} active symbols "
        f"(source={result.source_model}, version={result.model_version})"
    )
    return result


@activity.defn
def get_ppo_portfolio() -> AlpacaPortfolioResponse:
    """Fetch PPO Alpaca account portfolio."""
    logger.info("Fetching PPO portfolio from Alpaca...")
    with get_client() as client:
        response = client.get("/alpaca/portfolio", params={"account": "ppo"})
        response.raise_for_status()
    result = AlpacaPortfolioResponse(**response.json())
    logger.info(
        f"PPO portfolio: cash=${result.cash:.2f}, "
        f"{len(result.positions)} positions, "
        f"{result.open_orders_count} open orders"
    )
    return result


@activity.defn
def get_sac_portfolio() -> AlpacaPortfolioResponse:
    """Fetch SAC Alpaca account portfolio."""
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


def _submit_orders(
    account: str,
    orders: GenerateOrdersResponse | SkippedOrdersResponse,
) -> SubmitOrdersResponse | SkippedSubmitResponse:
    """Submit orders for the given account. Shared logic for PPO/SAC/HRP."""
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
def submit_orders_ppo(
    orders: GenerateOrdersResponse | SkippedOrdersResponse,
) -> SubmitOrdersResponse | SkippedSubmitResponse:
    """Submit PPO orders to Alpaca."""
    return _submit_orders("ppo", orders)


@activity.defn
def submit_orders_sac(
    orders: GenerateOrdersResponse | SkippedOrdersResponse,
) -> SubmitOrdersResponse | SkippedSubmitResponse:
    """Submit SAC orders to Alpaca."""
    return _submit_orders("sac", orders)


@activity.defn
def submit_orders_hrp(
    orders: GenerateOrdersResponse | SkippedOrdersResponse,
) -> SubmitOrdersResponse | SkippedSubmitResponse:
    """Submit HRP orders to Alpaca."""
    return _submit_orders("hrp", orders)


@activity.defn
def get_order_history_ppo(after_date: str) -> list[OrderHistoryItem]:
    """Fetch PPO order history from Alpaca."""
    logger.info(f"Fetching PPO order history after {after_date}...")
    with get_client() as client:
        response = client.get(
            "/alpaca/order-history", params={"account": "ppo", "after": after_date}
        )
        response.raise_for_status()
    result = [OrderHistoryItem(**o) for o in response.json()]
    logger.info(f"Got {len(result)} PPO orders from history")
    return result


@activity.defn
def get_order_history_sac(after_date: str) -> list[OrderHistoryItem]:
    """Fetch SAC order history from Alpaca."""
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
def check_order_statuses(account: str, client_order_ids: list[str]) -> list[dict]:
    """Check order statuses using /alpaca/order-history endpoint.

    Fetches recent order history and filters to the given client_order_ids.
    Returns list of {client_order_id, status, filled_qty, filled_avg_price}.
    """
    logger.info(
        f"Checking {len(client_order_ids)} order statuses for {account.upper()}..."
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
        f"Got {len(matched)}/{len(client_order_ids)} order statuses "
        f"for {account.upper()}"
    )
    return matched

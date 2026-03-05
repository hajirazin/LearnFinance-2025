"""Portfolio, order submission, order history, and order status activities."""

import logging

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
    """Check order statuses by client_order_id via brain_api.

    Returns list of {client_order_id, status, filled_qty, filled_avg_price}.
    """
    logger.info(
        f"Checking {len(client_order_ids)} order statuses for {account.upper()}..."
    )
    with get_client() as client:
        response = client.post(
            "/alpaca/check-orders",
            json={"account": account, "client_order_ids": client_order_ids},
        )
        response.raise_for_status()
    data = response.json()
    orders = data.get("orders", [])
    logger.info(f"Got {len(orders)} order statuses for {account.upper()}")
    return orders

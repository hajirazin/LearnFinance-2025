"""Portfolio, order submission, and order history tasks."""

from prefect import task
from prefect.logging import get_run_logger

from flows.models import (
    ActiveSymbolsResponse,
    AlpacaPortfolioResponse,
    GenerateOrdersResponse,
    OrderHistoryItem,
    SkippedOrdersResponse,
    SkippedSubmitResponse,
    SubmitOrdersResponse,
)
from flows.tasks.client import get_client

# =============================================================================
# Portfolio Tasks
# =============================================================================


@task(name="Get Active Symbols", retries=2, retry_delay_seconds=30)
def get_active_symbols() -> ActiveSymbolsResponse:
    """Fetch the active symbols from the current SAC model via brain_api."""
    logger = get_run_logger()
    logger.info("Fetching active symbols from SAC model...")

    with get_client() as client:
        response = client.get("/models/active-symbols")
        response.raise_for_status()
        data = response.json()

    result = ActiveSymbolsResponse(**data)
    logger.info(
        f"Got {len(result.symbols)} active symbols "
        f"(source={result.source_model}, version={result.model_version})"
    )
    return result


@task(name="Get PPO Portfolio", retries=2, retry_delay_seconds=30)
def get_ppo_portfolio() -> AlpacaPortfolioResponse:
    """Fetch PPO Alpaca account portfolio."""
    logger = get_run_logger()
    logger.info("Fetching PPO portfolio from Alpaca...")

    with get_client() as client:
        response = client.get("/alpaca/portfolio", params={"account": "ppo"})
        response.raise_for_status()
        data = response.json()

    result = AlpacaPortfolioResponse(**data)
    logger.info(
        f"PPO portfolio: cash=${result.cash:.2f}, "
        f"{len(result.positions)} positions, "
        f"{result.open_orders_count} open orders"
    )
    return result


@task(name="Get SAC Portfolio", retries=2, retry_delay_seconds=30)
def get_sac_portfolio() -> AlpacaPortfolioResponse:
    """Fetch SAC Alpaca account portfolio."""
    logger = get_run_logger()
    logger.info("Fetching SAC portfolio from Alpaca...")

    with get_client() as client:
        response = client.get("/alpaca/portfolio", params={"account": "sac"})
        response.raise_for_status()
        data = response.json()

    result = AlpacaPortfolioResponse(**data)
    logger.info(
        f"SAC portfolio: cash=${result.cash:.2f}, "
        f"{len(result.positions)} positions, "
        f"{result.open_orders_count} open orders"
    )
    return result


@task(name="Get HRP Portfolio", retries=2, retry_delay_seconds=30)
def get_hrp_portfolio() -> AlpacaPortfolioResponse:
    """Fetch HRP Alpaca account portfolio."""
    logger = get_run_logger()
    logger.info("Fetching HRP portfolio from Alpaca...")

    with get_client() as client:
        response = client.get("/alpaca/portfolio", params={"account": "hrp"})
        response.raise_for_status()
        data = response.json()

    result = AlpacaPortfolioResponse(**data)
    logger.info(
        f"HRP portfolio: cash=${result.cash:.2f}, "
        f"{len(result.positions)} positions, "
        f"{result.open_orders_count} open orders"
    )
    return result


# =============================================================================
# Order Submission Tasks
# =============================================================================


@task(name="Submit Orders PPO", retries=1, retry_delay_seconds=30)
def submit_orders_ppo(
    orders: GenerateOrdersResponse | SkippedOrdersResponse,
) -> SubmitOrdersResponse | SkippedSubmitResponse:
    """Submit PPO orders to Alpaca."""
    logger = get_run_logger()

    if isinstance(orders, SkippedOrdersResponse) or getattr(orders, "skipped", False):
        logger.info("PPO orders skipped")
        return SkippedSubmitResponse(account="ppo", skipped=True)

    if not orders.orders:
        logger.info("No PPO orders to submit")
        return SubmitOrdersResponse(
            account="ppo",
            orders_submitted=0,
            orders_failed=0,
            skipped=False,
            results=[],
        )

    logger.info(f"Submitting {len(orders.orders)} PPO orders...")

    with get_client() as client:
        response = client.post(
            "/alpaca/submit-orders",
            json={
                "account": "ppo",
                "orders": [o.model_dump() for o in orders.orders],
            },
        )
        response.raise_for_status()
        data = response.json()

    result = SubmitOrdersResponse(**data)
    logger.info(
        f"PPO orders: {result.orders_submitted} submitted, "
        f"{result.orders_failed} failed"
    )
    return result


@task(name="Submit Orders SAC", retries=1, retry_delay_seconds=30)
def submit_orders_sac(
    orders: GenerateOrdersResponse | SkippedOrdersResponse,
) -> SubmitOrdersResponse | SkippedSubmitResponse:
    """Submit SAC orders to Alpaca."""
    logger = get_run_logger()

    if isinstance(orders, SkippedOrdersResponse) or getattr(orders, "skipped", False):
        logger.info("SAC orders skipped")
        return SkippedSubmitResponse(account="sac", skipped=True)

    if not orders.orders:
        logger.info("No SAC orders to submit")
        return SubmitOrdersResponse(
            account="sac",
            orders_submitted=0,
            orders_failed=0,
            skipped=False,
            results=[],
        )

    logger.info(f"Submitting {len(orders.orders)} SAC orders...")

    with get_client() as client:
        response = client.post(
            "/alpaca/submit-orders",
            json={
                "account": "sac",
                "orders": [o.model_dump() for o in orders.orders],
            },
        )
        response.raise_for_status()
        data = response.json()

    result = SubmitOrdersResponse(**data)
    logger.info(
        f"SAC orders: {result.orders_submitted} submitted, "
        f"{result.orders_failed} failed"
    )
    return result


@task(name="Submit Orders HRP", retries=1, retry_delay_seconds=30)
def submit_orders_hrp(
    orders: GenerateOrdersResponse | SkippedOrdersResponse,
) -> SubmitOrdersResponse | SkippedSubmitResponse:
    """Submit HRP orders to Alpaca."""
    logger = get_run_logger()

    if isinstance(orders, SkippedOrdersResponse) or getattr(orders, "skipped", False):
        logger.info("HRP orders skipped")
        return SkippedSubmitResponse(account="hrp", skipped=True)

    if not orders.orders:
        logger.info("No HRP orders to submit")
        return SubmitOrdersResponse(
            account="hrp",
            orders_submitted=0,
            orders_failed=0,
            skipped=False,
            results=[],
        )

    logger.info(f"Submitting {len(orders.orders)} HRP orders...")

    with get_client() as client:
        response = client.post(
            "/alpaca/submit-orders",
            json={
                "account": "hrp",
                "orders": [o.model_dump() for o in orders.orders],
            },
        )
        response.raise_for_status()
        data = response.json()

    result = SubmitOrdersResponse(**data)
    logger.info(
        f"HRP orders: {result.orders_submitted} submitted, "
        f"{result.orders_failed} failed"
    )
    return result


# =============================================================================
# Order History Tasks
# =============================================================================


@task(name="Get Order History PPO", retries=2, retry_delay_seconds=30)
def get_order_history_ppo(after_date: str) -> list[OrderHistoryItem]:
    """Fetch PPO order history from Alpaca."""
    logger = get_run_logger()
    logger.info(f"Fetching PPO order history after {after_date}...")

    with get_client() as client:
        response = client.get(
            "/alpaca/order-history",
            params={"account": "ppo", "after": after_date},
        )
        response.raise_for_status()
        data = response.json()

    result = [OrderHistoryItem(**o) for o in data]
    logger.info(f"Got {len(result)} PPO orders from history")
    return result


@task(name="Get Order History SAC", retries=2, retry_delay_seconds=30)
def get_order_history_sac(after_date: str) -> list[OrderHistoryItem]:
    """Fetch SAC order history from Alpaca."""
    logger = get_run_logger()
    logger.info(f"Fetching SAC order history after {after_date}...")

    with get_client() as client:
        response = client.get(
            "/alpaca/order-history",
            params={"account": "sac", "after": after_date},
        )
        response.raise_for_status()
        data = response.json()

    result = [OrderHistoryItem(**o) for o in data]
    logger.info(f"Got {len(result)} SAC orders from history")
    return result

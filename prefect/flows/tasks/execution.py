"""Order generation and experience management tasks."""

from prefect import task
from prefect.logging import get_run_logger

from flows.models import (
    AlpacaPortfolioResponse,
    FundamentalsResponse,
    GenerateOrdersResponse,
    HRPAllocationResponse,
    LSTMInferenceResponse,
    NewsSignalResponse,
    OrderHistoryItem,
    PatchTSTInferenceResponse,
    PPOInferenceResponse,
    SACInferenceResponse,
    SkippedAllocation,
    SkippedOrdersResponse,
    StoreExperienceResponse,
    UpdateExecutionResponse,
)
from flows.tasks.client import get_client

# =============================================================================
# Order Generation Tasks
# =============================================================================


@task(name="Generate Orders PPO", retries=1, retry_delay_seconds=30)
def generate_orders_ppo(
    allocation: PPOInferenceResponse | SkippedAllocation,
    portfolio: AlpacaPortfolioResponse,
    run_id: str,
    attempt: int,
) -> GenerateOrdersResponse | SkippedOrdersResponse:
    """Generate orders for PPO allocation."""
    logger = get_run_logger()

    if isinstance(allocation, SkippedAllocation) or getattr(
        allocation, "skipped", False
    ):
        logger.info("PPO skipped - returning empty orders")
        return SkippedOrdersResponse(skipped=True, algorithm="ppo")

    logger.info("Generating PPO orders...")

    with get_client() as client:
        response = client.post(
            "/orders/generate",
            json={
                "target_weights": allocation.target_weights,
                "portfolio": {
                    "cash": portfolio.cash,
                    "positions": [p.model_dump() for p in portfolio.positions],
                },
                "run_id": run_id,
                "attempt": attempt,
                "algorithm": "ppo",
            },
        )
        response.raise_for_status()
        data = response.json()

    result = GenerateOrdersResponse(**data)
    logger.info(f"PPO orders: {result.summary.buys} buys, {result.summary.sells} sells")
    return result


@task(name="Generate Orders SAC", retries=1, retry_delay_seconds=30)
def generate_orders_sac(
    allocation: SACInferenceResponse | SkippedAllocation,
    portfolio: AlpacaPortfolioResponse,
    run_id: str,
    attempt: int,
) -> GenerateOrdersResponse | SkippedOrdersResponse:
    """Generate orders for SAC allocation."""
    logger = get_run_logger()

    if isinstance(allocation, SkippedAllocation) or getattr(
        allocation, "skipped", False
    ):
        logger.info("SAC skipped - returning empty orders")
        return SkippedOrdersResponse(skipped=True, algorithm="sac")

    logger.info("Generating SAC orders...")

    with get_client() as client:
        response = client.post(
            "/orders/generate",
            json={
                "target_weights": allocation.target_weights,
                "portfolio": {
                    "cash": portfolio.cash,
                    "positions": [p.model_dump() for p in portfolio.positions],
                },
                "run_id": run_id,
                "attempt": attempt,
                "algorithm": "sac",
            },
        )
        response.raise_for_status()
        data = response.json()

    result = GenerateOrdersResponse(**data)
    logger.info(f"SAC orders: {result.summary.buys} buys, {result.summary.sells} sells")
    return result


@task(name="Generate Orders HRP", retries=1, retry_delay_seconds=30)
def generate_orders_hrp(
    allocation: HRPAllocationResponse | SkippedAllocation,
    portfolio: AlpacaPortfolioResponse,
    run_id: str,
    attempt: int,
) -> GenerateOrdersResponse | SkippedOrdersResponse:
    """Generate orders for HRP allocation.

    Note: HRP returns percentage_weights (10.5 = 10.5%), which we convert
    to target_weights (0.105) before calling the orders endpoint.
    """
    logger = get_run_logger()

    if isinstance(allocation, SkippedAllocation) or getattr(
        allocation, "skipped", False
    ):
        logger.info("HRP skipped - returning empty orders")
        return SkippedOrdersResponse(skipped=True, algorithm="hrp")

    logger.info("Generating HRP orders...")

    # Convert percentage weights to decimal target weights
    target_weights = {
        sym: wt / 100 for sym, wt in allocation.percentage_weights.items()
    }

    with get_client() as client:
        response = client.post(
            "/orders/generate",
            json={
                "target_weights": target_weights,
                "portfolio": {
                    "cash": portfolio.cash,
                    "positions": [p.model_dump() for p in portfolio.positions],
                },
                "run_id": run_id,
                "attempt": attempt,
                "algorithm": "hrp",
            },
        )
        response.raise_for_status()
        data = response.json()

    result = GenerateOrdersResponse(**data)
    logger.info(f"HRP orders: {result.summary.buys} buys, {result.summary.sells} sells")
    return result


# =============================================================================
# Experience Storage Tasks
# =============================================================================


def _build_state_dict(
    portfolio: AlpacaPortfolioResponse,
    news: NewsSignalResponse,
    fundamentals: FundamentalsResponse,
    lstm: LSTMInferenceResponse,
    patchtst: PatchTSTInferenceResponse,
) -> dict:
    """Build state dict for experience storage."""
    total_value = portfolio.cash + sum(p.market_value for p in portfolio.positions)
    current_weights = {
        p.symbol: p.market_value / total_value for p in portfolio.positions
    }
    current_weights["CASH"] = portfolio.cash / total_value

    signals = {}
    for ns in news.per_symbol:
        signals[ns.symbol] = {"news_sentiment": ns.sentiment_score}
    for fs in fundamentals.per_symbol:
        if fs.ratios:
            if fs.symbol not in signals:
                signals[fs.symbol] = {}
            signals[fs.symbol].update(fs.ratios.model_dump())

    lstm_forecasts = {p.symbol: p.predicted_weekly_return_pct for p in lstm.predictions}
    patchtst_forecasts = {
        p.symbol: p.predicted_weekly_return_pct for p in patchtst.predictions
    }

    return {
        "signals": signals,
        "lstm_forecasts": lstm_forecasts,
        "patchtst_forecasts": patchtst_forecasts,
        "current_weights": current_weights,
    }


@task(name="Store Experience PPO", retries=1, retry_delay_seconds=30)
def store_experience_ppo(
    run_id: str,
    week_start: str,
    week_end: str,
    allocation: PPOInferenceResponse | SkippedAllocation,
    portfolio: AlpacaPortfolioResponse,
    news: NewsSignalResponse,
    fundamentals: FundamentalsResponse,
    lstm: LSTMInferenceResponse,
    patchtst: PatchTSTInferenceResponse,
) -> StoreExperienceResponse | None:
    """Store PPO experience for future reward labeling."""
    logger = get_run_logger()

    if isinstance(allocation, SkippedAllocation) or getattr(
        allocation, "skipped", False
    ):
        logger.info("PPO skipped - not storing experience")
        return None

    logger.info("Storing PPO experience...")

    state = _build_state_dict(portfolio, news, fundamentals, lstm, patchtst)

    with get_client() as client:
        response = client.post(
            "/experience/store",
            json={
                "run_id": run_id,
                "week_start": week_start,
                "week_end": week_end,
                "model_type": "ppo",
                "model_version": allocation.model_version,
                "state": state,
                "intended_action": allocation.target_weights,
                "intended_turnover": allocation.turnover,
            },
        )
        response.raise_for_status()
        data = response.json()

    result = StoreExperienceResponse(**data)
    logger.info(f"Stored PPO experience: {result.record_id}")
    return result


@task(name="Store Experience SAC", retries=1, retry_delay_seconds=30)
def store_experience_sac(
    run_id: str,
    week_start: str,
    week_end: str,
    allocation: SACInferenceResponse | SkippedAllocation,
    portfolio: AlpacaPortfolioResponse,
    news: NewsSignalResponse,
    fundamentals: FundamentalsResponse,
    lstm: LSTMInferenceResponse,
    patchtst: PatchTSTInferenceResponse,
) -> StoreExperienceResponse | None:
    """Store SAC experience for future reward labeling."""
    logger = get_run_logger()

    if isinstance(allocation, SkippedAllocation) or getattr(
        allocation, "skipped", False
    ):
        logger.info("SAC skipped - not storing experience")
        return None

    logger.info("Storing SAC experience...")

    state = _build_state_dict(portfolio, news, fundamentals, lstm, patchtst)

    with get_client() as client:
        response = client.post(
            "/experience/store",
            json={
                "run_id": run_id,
                "week_start": week_start,
                "week_end": week_end,
                "model_type": "sac",
                "model_version": allocation.model_version,
                "state": state,
                "intended_action": allocation.target_weights,
                "intended_turnover": allocation.turnover,
            },
        )
        response.raise_for_status()
        data = response.json()

    result = StoreExperienceResponse(**data)
    logger.info(f"Stored SAC experience: {result.record_id}")
    return result


# =============================================================================
# Execution Update Tasks
# =============================================================================


@task(name="Update Execution PPO", retries=1, retry_delay_seconds=30)
def update_execution_ppo(
    run_id: str,
    orders: GenerateOrdersResponse | SkippedOrdersResponse,
    history: list[OrderHistoryItem],
) -> UpdateExecutionResponse | None:
    """Update PPO experience with execution report."""
    logger = get_run_logger()

    if isinstance(orders, SkippedOrdersResponse) or getattr(orders, "skipped", False):
        logger.info("PPO skipped - not updating execution")
        return None

    if not orders.orders:
        logger.info("No PPO orders - not updating execution")
        return None

    logger.info("Updating PPO execution report...")

    # Convert to dicts for the API
    intended_orders = [
        {
            "symbol": o.symbol,
            "qty": o.qty,
            "side": o.side,
            "client_order_id": o.client_order_id,
        }
        for o in orders.orders
    ]
    executed_orders = [h.model_dump() for h in history]

    with get_client() as client:
        response = client.post(
            "/experience/update-execution",
            json={
                "run_id": run_id,
                "model_type": "ppo",
                "intended_orders": intended_orders,
                "executed_orders": executed_orders,
            },
        )
        response.raise_for_status()
        data = response.json()

    result = UpdateExecutionResponse(**data)
    logger.info(
        f"Updated PPO execution: filled={result.orders_filled}, "
        f"partial={result.orders_partial}, expired={result.orders_expired}"
    )
    return result


@task(name="Update Execution SAC", retries=1, retry_delay_seconds=30)
def update_execution_sac(
    run_id: str,
    orders: GenerateOrdersResponse | SkippedOrdersResponse,
    history: list[OrderHistoryItem],
) -> UpdateExecutionResponse | None:
    """Update SAC experience with execution report."""
    logger = get_run_logger()

    if isinstance(orders, SkippedOrdersResponse) or getattr(orders, "skipped", False):
        logger.info("SAC skipped - not updating execution")
        return None

    if not orders.orders:
        logger.info("No SAC orders - not updating execution")
        return None

    logger.info("Updating SAC execution report...")

    # Convert to dicts for the API
    intended_orders = [
        {
            "symbol": o.symbol,
            "qty": o.qty,
            "side": o.side,
            "client_order_id": o.client_order_id,
        }
        for o in orders.orders
    ]
    executed_orders = [h.model_dump() for h in history]

    with get_client() as client:
        response = client.post(
            "/experience/update-execution",
            json={
                "run_id": run_id,
                "model_type": "sac",
                "intended_orders": intended_orders,
                "executed_orders": executed_orders,
            },
        )
        response.raise_for_status()
        data = response.json()

    result = UpdateExecutionResponse(**data)
    logger.info(
        f"Updated SAC execution: filled={result.orders_filled}, "
        f"partial={result.orders_partial}, expired={result.orders_expired}"
    )
    return result

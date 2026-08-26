"""Order and experience activities for ppo_discovery."""

from __future__ import annotations

import logging

from temporalio import activity

from activities.client import get_client
from activities.portfolio import _submit_orders
from models.forecast_email import (
    AlpacaPortfolioResponse,
    GenerateOrdersResponse,
    OrderHistoryItem,
    SkippedOrdersResponse,
    SkippedSubmitResponse,
    StoreExperienceResponse,
    SubmitOrdersResponse,
    UpdateExecutionResponse,
)
from models.ppo_discovery import PPOInferenceResponse

logger = logging.getLogger(__name__)


@activity.defn
def generate_orders_ppo_discovery(
    allocation: PPOInferenceResponse,
    portfolio: AlpacaPortfolioResponse,
    run_id: str,
    attempt: int,
    order_side: str = "all",
) -> GenerateOrdersResponse | SkippedOrdersResponse:
    if allocation.skipped:
        return SkippedOrdersResponse(skipped=True, algorithm="ppo_discovery")
    with get_client() as client:
        response = client.post(
            "/orders/generate",
            json={
                "target_weights": allocation.percentage_weights,
                "portfolio": {
                    "cash": portfolio.cash,
                    "currency": getattr(portfolio, "currency", "USD"),
                    "positions": [p.model_dump() for p in portfolio.positions],
                },
                "run_id": run_id,
                "attempt": attempt,
                "algorithm": "ppo_discovery",
                "order_side": order_side,
            },
        )
        response.raise_for_status()
    return GenerateOrdersResponse(**response.json())


@activity.defn
def submit_orders_ppo_discovery(
    orders: GenerateOrdersResponse | SkippedOrdersResponse,
) -> SubmitOrdersResponse | SkippedSubmitResponse:
    return _submit_orders("ppo_discovery", orders)


@activity.defn
def store_experience_ppo_discovery(
    run_id: str,
    week_start: str,
    week_end: str,
    allocation: PPOInferenceResponse,
    state: dict,
) -> StoreExperienceResponse | None:
    if allocation.skipped:
        return None
    with get_client() as client:
        response = client.post(
            "/experience/store",
            json={
                "run_id": run_id,
                "week_start": week_start,
                "week_end": week_end,
                "model_type": "ppo_discovery",
                "model_version": allocation.model_version,
                "universe": "halal_new",
                "state": {**state, "digest": allocation.state_digest},
                "state_digest": allocation.state_digest,
                "intended_action": allocation.percentage_weights,
            },
        )
        response.raise_for_status()
    return StoreExperienceResponse(**response.json())


def _portfolio_weights_and_nav(
    portfolio: AlpacaPortfolioResponse,
) -> tuple[dict[str, float], float]:
    nav = float(portfolio.cash) + sum(p.market_value for p in portfolio.positions)
    if nav <= 0:
        return {"CASH": 1.0}, nav
    weights = {p.symbol: p.market_value / nav for p in portfolio.positions}
    weights["CASH"] = float(portfolio.cash) / nav
    return weights, nav


@activity.defn
def get_order_history_ppo_discovery(after_date: str) -> list[OrderHistoryItem]:
    with get_client() as client:
        response = client.get(
            "/alpaca/order-history",
            params={"account": "ppo_discovery", "after": after_date},
        )
        response.raise_for_status()
    return [OrderHistoryItem(**row) for row in response.json()]


@activity.defn
def update_execution_ppo_discovery(
    run_id: str,
    orders: GenerateOrdersResponse | SkippedOrdersResponse,
    history: list[OrderHistoryItem],
    post_trade_portfolio: AlpacaPortfolioResponse,
) -> UpdateExecutionResponse | None:
    if isinstance(orders, SkippedOrdersResponse) or getattr(orders, "skipped", False):
        return None
    if not orders.orders:
        return None
    actual_weights, nav_usd = _portfolio_weights_and_nav(post_trade_portfolio)
    with get_client() as client:
        response = client.post(
            "/experience/update-execution",
            json={
                "run_id": run_id,
                "model_type": "ppo_discovery",
                "intended_orders": [
                    {
                        "symbol": order.symbol,
                        "qty": order.qty,
                        "side": order.side,
                        "client_order_id": order.client_order_id,
                    }
                    for order in orders.orders
                ],
                "executed_orders": [row.model_dump() for row in history],
                "actual_weights": actual_weights,
                "nav_usd": nav_usd,
            },
        )
        response.raise_for_status()
    return UpdateExecutionResponse(**response.json())


@activity.defn
def label_ppo_discovery_experience(run_id: str | None = None) -> dict:
    with get_client() as client:
        payload = {} if not run_id else {"run_id": run_id}
        response = client.post("/experience/label/ppo-discovery", json=payload)
        response.raise_for_status()
    return response.json()

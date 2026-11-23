"""Verify sell rejection handling through the SAC halal workflow."""

import pytest
from temporalio import activity

from models import (
    GenerateOrdersResponse,
    OrderModel,
    OrderSubmitResult,
    PortfolioResponse,
    SubmitOrdersResponse,
)
from tests.harness import worker_with_activities
from tests.test_us_sac_halal_allocation import (
    _make_orders,
    _make_sac_halal_activities,
)
from workflows.us_sac_halal_allocation import USSACHalalAllocationWorkflow


async def _run_sell_case(patchtst_resp, news_resp, mixed):
    rejected = "paper:halal:2026-05-04:attempt-1:TSLA:SELL"
    accepted = "paper:halal:2026-05-04:attempt-1:AAPL:SELL"
    captured = {}
    sequence = []
    requested_ids = []
    initial = PortfolioResponse(cash=1000, positions=[], open_orders_count=0)
    refreshed = initial.model_copy(update={"cash": 1500 if mixed else 1000})
    stubs = _make_sac_halal_activities(
        n_symbols=10,
        sac_portfolio=initial,
        patchtst_resp=patchtst_resp,
        news_resp=news_resp,
        captured_calls=captured,
    )

    @activity.defn(name="generate_sell_orders_sac")
    def generate_sells(allocation, portfolio, run_id, attempt, algorithm):
        sells = [
            OrderModel(
                client_order_id=order_id,
                symbol=symbol,
                side="sell",
                qty=qty,
                type="market",
                time_in_force="day",
            )
            for order_id, symbol, qty in [(rejected, "TSLA", 3.1622)]
            + ([(accepted, "AAPL", 2)] if mixed else [])
        ]
        template = _make_orders()
        return GenerateOrdersResponse(
            orders=sells,
            summary=template.summary.model_copy(
                update={"buys": 0, "sells": len(sells)}
            ),
            prices_used={"TSLA": 385, "AAPL": 250},
        )

    @activity.defn(name="submit_orders_ibkr_sac_halal")
    def submit(orders: GenerateOrdersResponse) -> SubmitOrdersResponse:
        side = orders.orders[0].side
        sequence.append(f"submit_{side}")
        results = [
            OrderSubmitResult(
                client_order_id=order.client_order_id,
                symbol=order.symbol,
                id=None if order.client_order_id == rejected else "order-1",
                status="rejected" if order.client_order_id == rejected else "Submitted",
            )
            for order in orders.orders
        ]
        failed = int(side == "sell")
        return SubmitOrdersResponse(
            account="sac_halal",
            orders_submitted=len(results) - failed,
            orders_failed=failed,
            results=results,
        )

    @activity.defn(name="check_order_statuses_ibkr")
    def check_statuses(account, client_order_ids):
        assert account == "sac_halal"
        requested_ids.append(client_order_ids)
        sequence.append("poll")
        accepted_status = "PendingCancel" if len(requested_ids) == 1 else "canceled"
        return [
            {
                "client_order_id": order_id,
                "status": "rejected" if order_id == rejected else accepted_status,
            }
            for order_id in client_order_ids
        ]

    @activity.defn(name="get_ibkr_sac_halal_portfolio")
    def get_portfolio() -> PortfolioResponse:
        sequence.append("portfolio")
        return initial if len(sequence) == 1 else refreshed

    @activity.defn(name="generate_buy_orders_sac")
    def generate_buys(allocation, portfolio: PortfolioResponse, run_id, attempt, algo):
        sequence.append("generate_buys")
        captured["buy_cash"] = portfolio.cash
        return _make_orders()

    overrides = {
        "mock_generate_sell_orders_sac": generate_sells,
        "mock_submit_orders_ibkr_sac_halal": submit,
        "mock_check_order_statuses_ibkr": check_statuses,
        "mock_get_ibkr_sac_halal_portfolio": get_portfolio,
        "mock_generate_buy_orders_sac": generate_buys,
    }
    activities = [overrides.get(stub.__name__, stub) for stub in stubs]
    async with worker_with_activities(
        [USSACHalalAllocationWorkflow], activities
    ) as env:
        handle = await env.client.start_workflow(
            USSACHalalAllocationWorkflow.run,
            id="test-sac-halal-rejection",
            task_queue="test-queue",
        )
        await handle.result()
        history = await handle.fetch_history()
    return sequence, requested_ids, captured, accepted, history


@pytest.mark.parametrize("mixed", [False, True])
async def test_only_working_sells_are_polled_before_fresh_buy_pricing(
    patchtst_resp, news_resp, mixed
):
    sequence, requested_ids, captured, accepted, _ = await _run_sell_case(
        patchtst_resp, news_resp, mixed
    )

    assert requested_ids == ([[accepted], [accepted]] if mixed else [])
    assert sequence == [
        "portfolio",
        "submit_sell",
        *(["poll", "poll"] if mixed else []),
        "portfolio",
        "generate_buys",
        "submit_buy",
        "portfolio",
    ]
    assert captured["buy_cash"] == (1500 if mixed else 1000)
    assert captured["email_order_details"][0]["submission_status"] == "rejected"

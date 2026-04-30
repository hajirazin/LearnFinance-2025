"""Sell-wait-buy interaction test for ``USAlphaHRPWorkflow``.

Validates that when generated orders contain both sells and buys, the
workflow submits sells first, polls until they reach a terminal state,
then submits buys.
"""

from __future__ import annotations

import pytest

from tests.harness import make_us_alpha_hrp_activities, worker_with_activities
from workflows.us_alpha_hrp import USAlphaHRPWorkflow


class TestUSAlphaHRPSellWaitBuy:
    @pytest.mark.asyncio
    async def test_terminal_sells_then_buys(
        self,
        universe_data,
        hrp_portfolio_no_open,
        patchtst_scores,
        stage2_alloc,
        sticky_cold_start,
        record_final_resp,
        alpha_orders_sell_and_buy,
        submit_resp,
        summary_resp,
        email_resp,
    ):
        submit_calls: list = []
        status_call_count = {"n": 0}

        def status_fn(account, client_order_ids):
            status_call_count["n"] += 1
            return [
                {"client_order_id": cid, "status": "filled"} for cid in client_order_ids
            ]

        activities = make_us_alpha_hrp_activities(
            universe_data=universe_data,
            hrp_portfolio=hrp_portfolio_no_open,
            scores=patchtst_scores,
            stage2=stage2_alloc,
            sticky=sticky_cold_start,
            record_final=record_final_resp,
            orders=alpha_orders_sell_and_buy,
            submit_resp=submit_resp,
            summary=summary_resp,
            email=email_resp,
            submit_calls=submit_calls,
            check_order_statuses_fn=status_fn,
        )

        async with worker_with_activities([USAlphaHRPWorkflow], activities) as env:
            result = await env.client.execute_workflow(
                USAlphaHRPWorkflow.run,
                id="test-us-alpha-hrp-sellwaitbuy",
                task_queue="test-queue",
            )

        assert result["skipped"] is False
        assert len(submit_calls) == 2

        def _orders_of(call):
            if hasattr(call, "orders"):
                return [
                    o.side if hasattr(o, "side") else o["side"] for o in call.orders
                ]
            return [o["side"] for o in call["orders"]]

        first_sides = set(_orders_of(submit_calls[0]))
        second_sides = set(_orders_of(submit_calls[1]))
        assert first_sides == {"sell"}
        assert second_sides == {"buy"}
        assert status_call_count["n"] == 1

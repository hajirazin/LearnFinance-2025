"""Sell-wait-buy test for the SAC-only ``USWeeklyAllocationWorkflow``.

Validates that when SAC produces both sells and buys, the workflow
polls the ``sac`` account through to a terminal state before submitting
the buys.
"""

from __future__ import annotations

import pytest

from tests.harness import make_sac_only_activities, worker_with_activities
from workflows.us_weekly_allocation import USWeeklyAllocationWorkflow


class TestUSWeeklyAllocationSACSellWaitBuy:
    @pytest.mark.asyncio
    async def test_sac_sell_wait_buy_completes(
        self,
        active_symbols,
        sac_portfolio_no_open,
        lstm_resp,
        patchtst_resp,
        news_resp,
        fundamentals_resp,
        sac_alloc,
        sell_and_buy_orders,
        sac_submit_resp,
        sac_summary_resp,
        sac_email_resp,
    ):
        poll_count = {"n": 0}

        def staggered_check(account, client_order_ids):
            assert account == "sac", (
                f"SAC-only workflow must not poll account {account!r}"
            )
            poll_count["n"] += 1
            if poll_count["n"] <= 1:
                return [
                    {"client_order_id": cid, "status": "pending_new"}
                    for cid in client_order_ids
                ]
            return [
                {"client_order_id": cid, "status": "filled"} for cid in client_order_ids
            ]

        forbidden_calls: list[str] = []
        activities = make_sac_only_activities(
            active_symbols=active_symbols,
            sac_portfolio=sac_portfolio_no_open,
            fundamentals_resp=fundamentals_resp,
            news_resp=news_resp,
            lstm_resp=lstm_resp,
            patchtst_resp=patchtst_resp,
            sac_alloc=sac_alloc,
            sac_orders=sell_and_buy_orders,
            sac_submit_resp=sac_submit_resp,
            summary_resp=sac_summary_resp,
            email_resp=sac_email_resp,
            forbidden_calls=forbidden_calls,
            check_order_statuses_fn=staggered_check,
        )

        async with worker_with_activities(
            [USWeeklyAllocationWorkflow], activities
        ) as env:
            result = await env.client.execute_workflow(
                USWeeklyAllocationWorkflow.run,
                id="test-us-sac-sell-wait-buy",
                task_queue="test-queue",
            )

        assert result["sac"]["orders_submitted"] > 0
        assert result["email"]["is_success"] is True
        assert poll_count["n"] >= 2
        assert forbidden_calls == []

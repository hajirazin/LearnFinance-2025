"""Skip-path test for ``USAlphaHRPWorkflow``.

If the hrp Alpaca account has open orders, the workflow must short-circuit
without scoring, sticky selection, HRP, or order submission, and only send
a "skipped" email.
"""

from __future__ import annotations

import pytest

from tests.harness import make_us_alpha_hrp_activities, worker_with_activities
from workflows.us_alpha_hrp import USAlphaHRPWorkflow


class TestUSAlphaHRPSkipPath:
    @pytest.mark.asyncio
    async def test_skip_when_open_orders(
        self,
        universe_data,
        hrp_portfolio_with_open,
        patchtst_scores,
        stage2_alloc,
        sticky_cold_start,
        record_final_resp,
        alpha_orders_buys_only,
        submit_resp,
        summary_resp,
        email_resp,
    ):
        score_calls: list[dict] = []
        select_calls: list[dict] = []
        hrp_calls: list[dict] = []
        record_final_calls: list[dict] = []
        submit_calls: list = []

        activities = make_us_alpha_hrp_activities(
            universe_data=universe_data,
            hrp_portfolio=hrp_portfolio_with_open,
            scores=patchtst_scores,
            stage2=stage2_alloc,
            sticky=sticky_cold_start,
            record_final=record_final_resp,
            orders=alpha_orders_buys_only,
            submit_resp=submit_resp,
            summary=summary_resp,
            email=email_resp,
            score_calls=score_calls,
            select_calls=select_calls,
            hrp_calls=hrp_calls,
            record_final_calls=record_final_calls,
            submit_calls=submit_calls,
        )

        async with worker_with_activities([USAlphaHRPWorkflow], activities) as env:
            result = await env.client.execute_workflow(
                USAlphaHRPWorkflow.run,
                id="test-us-alpha-hrp-skip",
                task_queue="test-queue",
            )

        assert result["skipped"] is True
        assert result["skip_reason"] == "open_orders"
        assert result["email"]["is_success"] is True

        assert score_calls == []
        assert select_calls == []
        assert hrp_calls == []
        assert record_final_calls == []
        assert submit_calls == []

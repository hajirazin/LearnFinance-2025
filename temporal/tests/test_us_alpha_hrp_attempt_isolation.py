"""Attempt-isolation test for ``USAlphaHRPWorkflow``.

The Alpha-HRP workflow trades through the dedicated ``hrp`` Alpaca
account, so attempt resolution must be scoped to that account to keep
attempts isolated from the other US workflows.
"""

from __future__ import annotations

import pytest

from tests.harness import make_us_alpha_hrp_activities, worker_with_activities
from workflows.us_alpha_hrp import USAlphaHRPWorkflow


class TestUSAlphaHRPAttemptIsolation:
    @pytest.mark.asyncio
    async def test_resolve_next_attempt_called_with_hrp_account(
        self,
        universe_data,
        hrp_portfolio_no_open,
        patchtst_scores,
        stage2_alloc,
        sticky_cold_start,
        record_final_resp,
        alpha_orders_buys_only,
        submit_resp,
        summary_resp,
        email_resp,
    ):
        resolve_calls: list[dict] = []

        activities = make_us_alpha_hrp_activities(
            universe_data=universe_data,
            hrp_portfolio=hrp_portfolio_no_open,
            scores=patchtst_scores,
            stage2=stage2_alloc,
            sticky=sticky_cold_start,
            record_final=record_final_resp,
            orders=alpha_orders_buys_only,
            submit_resp=submit_resp,
            summary=summary_resp,
            email=email_resp,
            resolve_calls=resolve_calls,
        )

        async with worker_with_activities([USAlphaHRPWorkflow], activities) as env:
            await env.client.execute_workflow(
                USAlphaHRPWorkflow.run,
                id="test-us-alpha-hrp-attempt-isolation",
                task_queue="test-queue",
            )

        assert len(resolve_calls) == 1
        assert resolve_calls[0]["accounts"] == ["hrp"], (
            "USAlphaHRPWorkflow trades through the hrp Alpaca account; "
            "attempt resolution must be scoped to that account."
        )

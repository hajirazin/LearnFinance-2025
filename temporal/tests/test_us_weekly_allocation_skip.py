"""Skip-path test for the SAC-only ``USWeeklyAllocationWorkflow``.

When the SAC Alpaca account has open orders, the workflow must skip the
SAC allocator and produce an empty SAC orders payload while still
sending the weekly email.
"""

from __future__ import annotations

import pytest

from models import SkippedOrdersResponse
from tests.harness import make_sac_only_activities, worker_with_activities
from workflows.us_weekly_allocation import USWeeklyAllocationWorkflow


class TestUSWeeklyAllocationSACSkip:
    @pytest.mark.asyncio
    async def test_skip_sac_when_open_orders(
        self,
        active_symbols,
        sac_portfolio_with_open,
        lstm_resp,
        patchtst_resp,
        news_resp,
        fundamentals_resp,
        sac_alloc,
        sac_submit_resp,
        sac_summary_resp,
        sac_email_resp,
    ):
        forbidden_calls: list[str] = []
        activities = make_sac_only_activities(
            active_symbols=active_symbols,
            sac_portfolio=sac_portfolio_with_open,
            fundamentals_resp=fundamentals_resp,
            news_resp=news_resp,
            lstm_resp=lstm_resp,
            patchtst_resp=patchtst_resp,
            sac_alloc=sac_alloc,
            sac_orders=SkippedOrdersResponse(skipped=True, algorithm="sac"),
            sac_submit_resp=sac_submit_resp,
            summary_resp=sac_summary_resp,
            email_resp=sac_email_resp,
            forbidden_calls=forbidden_calls,
        )

        async with worker_with_activities(
            [USWeeklyAllocationWorkflow], activities
        ) as env:
            result = await env.client.execute_workflow(
                USWeeklyAllocationWorkflow.run,
                id="test-us-inference-sac-skip",
                task_queue="test-queue",
            )

        assert "SAC" in result["skipped_algorithms"]
        assert result["sac"]["skipped"] is True
        assert result["email"]["is_success"] is True
        assert forbidden_calls == []

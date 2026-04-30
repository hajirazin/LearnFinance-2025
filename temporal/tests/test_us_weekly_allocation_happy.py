"""Happy-path test for the SAC-only ``USWeeklyAllocationWorkflow``.

Asserts that the legacy HRP allocator/orders/submit activities are not
invoked and that the email payload still carries an empty ``hrp``
placeholder for schema stability.
"""

from __future__ import annotations

import pytest

from tests.harness import make_sac_only_activities, worker_with_activities
from workflows.us_weekly_allocation import USWeeklyAllocationWorkflow


class TestUSWeeklyAllocationSACOnlyHappyPath:
    @pytest.mark.asyncio
    async def test_sac_only_pipeline_runs_without_hrp_activities(
        self,
        active_symbols,
        sac_portfolio_no_open,
        lstm_resp,
        patchtst_resp,
        news_resp,
        fundamentals_resp,
        sac_alloc,
        buy_only_orders,
        sac_submit_resp,
        sac_summary_resp,
        sac_email_resp,
    ):
        forbidden_calls: list[str] = []
        summary_calls: list[dict] = []
        email_calls: list[dict] = []

        activities = make_sac_only_activities(
            active_symbols=active_symbols,
            sac_portfolio=sac_portfolio_no_open,
            fundamentals_resp=fundamentals_resp,
            news_resp=news_resp,
            lstm_resp=lstm_resp,
            patchtst_resp=patchtst_resp,
            sac_alloc=sac_alloc,
            sac_orders=buy_only_orders,
            sac_submit_resp=sac_submit_resp,
            summary_resp=sac_summary_resp,
            email_resp=sac_email_resp,
            forbidden_calls=forbidden_calls,
            summary_calls=summary_calls,
            email_calls=email_calls,
        )

        async with worker_with_activities(
            [USWeeklyAllocationWorkflow], activities
        ) as env:
            result = await env.client.execute_workflow(
                USWeeklyAllocationWorkflow.run,
                id="test-us-inference-sac-only",
                task_queue="test-queue",
            )

        assert result["symbols_count"] == 15
        assert result["skipped_algorithms"] == []
        assert result["sac"]["skipped"] is False
        assert result["sac"]["orders_submitted"] > 0
        assert result["email"]["is_success"] is True

        assert "hrp" not in result

        assert forbidden_calls == [], (
            "USWeeklyAllocationWorkflow is SAC-only post-refactor; "
            f"observed retired calls: {forbidden_calls}"
        )

        assert summary_calls and summary_calls[0]["hrp_symbols_used"] == 0
        assert email_calls and email_calls[0]["hrp_symbols_used"] == 0
        assert email_calls[0]["hrp_submit_skipped"] is True

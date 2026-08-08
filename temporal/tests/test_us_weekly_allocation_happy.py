"""Happy-path test for the SAC-only ``USWeeklyAllocationWorkflow``.

Asserts that the legacy HRP allocator/orders/submit activities are not
invoked and that the SAC-only weekly summary/email activities are
called with SAC-only positional args (no HRP placeholders).
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
        store_experience_calls: list[dict] = []
        update_execution_calls: list[dict] = []

        activities = make_sac_only_activities(
            active_symbols=active_symbols,
            sac_portfolio=sac_portfolio_no_open,
            fundamentals_resp=fundamentals_resp,
            news_resp=news_resp,
            patchtst_resp=patchtst_resp,
            sac_alloc=sac_alloc,
            sac_orders=buy_only_orders,
            sac_submit_resp=sac_submit_resp,
            summary_resp=sac_summary_resp,
            email_resp=sac_email_resp,
            forbidden_calls=forbidden_calls,
            summary_calls=summary_calls,
            email_calls=email_calls,
            store_experience_calls=store_experience_calls,
            update_execution_calls=update_execution_calls,
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

        assert summary_calls and summary_calls[0]["sac_skipped"] is False
        assert email_calls and email_calls[0]["sac_skipped"] is False
        assert email_calls[0]["sac_submit_skipped"] is False

        # Universe must be plumbed through explicitly per AGENTS.md
        # (no silent fallbacks). The legacy weekly workflow is the
        # halal_filtered A/B variant.
        assert summary_calls[0]["universe"] == "halal_filtered"
        assert email_calls[0]["universe"] == "halal_filtered"

        # Universe must also be persisted onto the experience record so
        # /experience/label/sac routes this record to the sac (not
        # sac_halal) Alpaca account at label time. Without this the
        # labeller would default to the legacy sac account by accident
        # for any halal record sharing model_type='sac'.
        assert store_experience_calls
        assert store_experience_calls[0]["universe"] == "halal_filtered"
        assert store_experience_calls[0]["decision_state"] == sac_alloc.decision_state
        assert store_experience_calls[0]["state_digest"] == sac_alloc.state_digest

        # Post-trade portfolio MUST flow into update_execution_sac so
        # the labeller never falls back to a live Alpaca query for
        # actual_weights.
        assert update_execution_calls
        assert update_execution_calls[0]["has_post_trade_portfolio"] is True

        # Per the email-enhancement plan, the per-order detail table
        # plus the "Going Into This Week" prior-allocation snapshot
        # must reach send_weekly_email. Both are US-only inputs and
        # built inside the workflow from generated orders + the live
        # broker portfolio respectively; if they aren't threaded the
        # email body silently regresses to the old summary-only form.
        def _attr(obj, name):
            return obj[name] if isinstance(obj, dict) else getattr(obj, name)

        assert email_calls[0]["order_details"] is not None
        assert len(email_calls[0]["order_details"]) >= 1
        first_detail = email_calls[0]["order_details"][0]
        assert _attr(first_detail, "symbol")
        assert _attr(first_detail, "side") in {"buy", "sell"}
        assert _attr(first_detail, "stop_loss_reason") in {
            "atr14",
            "atr_unavailable",
            "sell_no_stop",
        }

        prior = email_calls[0]["prior_allocation"]
        assert prior is not None
        assert _attr(prior, "source_label")  # US live-broker label, never empty
        assert isinstance(_attr(prior, "weights"), dict)

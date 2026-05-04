"""Happy-path tests for ``USAlphaHRPWorkflow``.

Covers cold-start, stable-week (all kept), and eviction-week scenarios.
Skip / sell-wait-buy / attempt-isolation live in sibling files so each
file stays focused and well under the 600-line policy limit.
"""

from __future__ import annotations

import pytest

from tests.harness import make_us_alpha_hrp_activities, worker_with_activities
from workflows.us_alpha_hrp import USAlphaHRPWorkflow


class TestUSAlphaHRPHappyPath:
    @pytest.mark.asyncio
    async def test_full_workflow_cold_start(
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
        score_calls: list[dict] = []
        select_calls: list[dict] = []
        hrp_calls: list[dict] = []
        record_final_calls: list[dict] = []

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
            score_calls=score_calls,
            select_calls=select_calls,
            hrp_calls=hrp_calls,
            record_final_calls=record_final_calls,
        )

        async with worker_with_activities([USAlphaHRPWorkflow], activities) as env:
            result = await env.client.execute_workflow(
                USAlphaHRPWorkflow.run,
                id="test-us-alpha-hrp-cold-start",
                task_queue="test-queue",
            )

        assert result["skipped"] is False
        assert result["universe_symbols"] == 20
        assert result["stage1_predicted_count"] == 20
        assert result["model_version"] == "v2026-04-26-abc"
        assert result["top_n"] == 15
        assert result["hold_threshold"] == 30
        assert result["kept_count"] == 0
        assert result["fillers_count"] == 15
        assert result["previous_year_week_used"] is None
        assert result["stage2_symbols_used"] == 15
        assert len(result["selected_symbols"]) == 15
        assert result["email"]["is_success"] is True

        assert len(score_calls) == 1
        assert score_calls[0]["symbols"] == [
            s["symbol"] for s in universe_data["stocks"]
        ]
        assert score_calls[0]["min_predictions"] == 15

        assert len(select_calls) == 1
        assert select_calls[0]["universe"] == "halal_new_alpha"
        assert select_calls[0]["top_n"] == 15
        assert select_calls[0]["hold_threshold"] == 30
        assert select_calls[0]["scores_count"] == 20

        assert len(hrp_calls) == 1
        assert hrp_calls[0]["lookback_days"] == 252
        assert hrp_calls[0]["symbols"] == sticky_cold_start.selected

        assert len(record_final_calls) == 1
        assert record_final_calls[0]["universe"] == "halal_new_alpha"
        assert record_final_calls[0]["year_week"] == select_calls[0]["year_week"]
        assert record_final_calls[0]["n_weights"] == 15

    @pytest.mark.asyncio
    async def test_email_receives_order_details_and_prior_allocation(
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
        """Workflow plumbing: email activity must see the new fields."""
        email_calls: list[dict] = []
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
            email_calls=email_calls,
        )

        async with worker_with_activities([USAlphaHRPWorkflow], activities) as env:
            await env.client.execute_workflow(
                USAlphaHRPWorkflow.run,
                id="test-us-alpha-hrp-email-payload",
                task_queue="test-queue",
            )

        # The harness binds positional args/kwargs against the real
        # send_us_alpha_hrp_email signature, so we assert by name and
        # don't have to track the parameter order. New optional kwargs
        # added to the activity in the future won't shift any indices.
        assert len(email_calls) == 1
        captured = email_calls[0]
        order_details = captured["order_details"]
        prior_allocation = captured["prior_allocation"]

        # The Temporal Pydantic data converter rehydrates the workflow-
        # side payload as the real model when a typed parameter exists,
        # but the mock declares ``*args`` so values come through as
        # dicts. Normalize via getattr/dict access either way.
        def _attr(obj, name):
            return obj[name] if isinstance(obj, dict) else getattr(obj, name)

        assert isinstance(order_details, list)
        assert len(order_details) >= 1
        first = order_details[0]
        assert _attr(first, "stop_loss_reason") in {
            "atr14",
            "atr_unavailable",
            "sell_no_stop",
        }
        assert prior_allocation is not None
        assert "live Alpaca account: hrp" in _attr(prior_allocation, "source_label")
        assert _attr(prior_allocation, "weights"), "prior allocation must not be empty"

    @pytest.mark.asyncio
    async def test_stable_week_all_kept(
        self,
        universe_data,
        hrp_portfolio_no_open,
        patchtst_scores,
        stage2_alloc,
        sticky_stable,
        record_final_resp,
        alpha_orders_buys_only,
        submit_resp,
        summary_resp,
        email_resp,
    ):
        activities = make_us_alpha_hrp_activities(
            universe_data=universe_data,
            hrp_portfolio=hrp_portfolio_no_open,
            scores=patchtst_scores,
            stage2=stage2_alloc,
            sticky=sticky_stable,
            record_final=record_final_resp,
            orders=alpha_orders_buys_only,
            submit_resp=submit_resp,
            summary=summary_resp,
            email=email_resp,
        )

        async with worker_with_activities([USAlphaHRPWorkflow], activities) as env:
            result = await env.client.execute_workflow(
                USAlphaHRPWorkflow.run,
                id="test-us-alpha-hrp-stable",
                task_queue="test-queue",
            )

        assert result["kept_count"] == 15
        assert result["fillers_count"] == 0
        assert result["previous_year_week_used"] == "202617"

    @pytest.mark.asyncio
    async def test_eviction_week(
        self,
        universe_data,
        hrp_portfolio_no_open,
        patchtst_scores,
        stage2_alloc,
        sticky_eviction,
        record_final_resp,
        alpha_orders_buys_only,
        submit_resp,
        summary_resp,
        email_resp,
    ):
        activities = make_us_alpha_hrp_activities(
            universe_data=universe_data,
            hrp_portfolio=hrp_portfolio_no_open,
            scores=patchtst_scores,
            stage2=stage2_alloc,
            sticky=sticky_eviction,
            record_final=record_final_resp,
            orders=alpha_orders_buys_only,
            submit_resp=submit_resp,
            summary=summary_resp,
            email=email_resp,
        )

        async with worker_with_activities([USAlphaHRPWorkflow], activities) as env:
            result = await env.client.execute_workflow(
                USAlphaHRPWorkflow.run,
                id="test-us-alpha-hrp-eviction",
                task_queue="test-queue",
            )

        assert result["kept_count"] == 13
        assert result["fillers_count"] == 2

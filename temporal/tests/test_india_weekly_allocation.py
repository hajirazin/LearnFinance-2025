"""Tests for India weekly allocation Temporal workflow.

Tests:
- Full workflow execution with mocked activities
- Sequential dependency (failure propagation)
- Correct date calculations
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pytest
from temporalio import activity
from temporalio.client import WorkflowFailureError
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from models import (
    HRPAllocationResponse,
    WeeklyReportEmailResponse,
    WeeklySummaryResponse,
)
from workflows.india_weekly_allocation import IndiaWeeklyAllocationWorkflow


@pytest.fixture
def mock_universe_data():
    return {
        "stocks": [
            {"symbol": "RELIANCE.NS", "predicted_weekly_return_pct": 3.5, "rank": 1},
            {"symbol": "TCS.NS", "predicted_weekly_return_pct": 3.2, "rank": 2},
            {"symbol": "INFY.NS", "predicted_weekly_return_pct": 2.8, "rank": 3},
        ],
        "total_candidates": 180,
        "total_universe": 210,
        "top_n": 15,
        "selection_method": "patchtst_forecast",
    }


@pytest.fixture
def mock_hrp():
    return HRPAllocationResponse(
        percentage_weights={"RELIANCE.NS": 25.0, "TCS.NS": 20.0, "INFY.NS": 15.0},
        symbols_used=3,
        symbols_excluded=[],
        as_of_date="2026-03-02",
    )


@pytest.fixture
def mock_summary():
    return WeeklySummaryResponse(
        summary={
            "para_1_portfolio_overview": "HRP distributed weights across 3 stocks."
        },
        provider="openai",
        model_used="gpt-4o-mini",
        tokens_used=350,
    )


@pytest.fixture
def mock_email():
    return WeeklyReportEmailResponse(
        is_success=True,
        subject="India Alpha-HRP Portfolio Analysis (2026-03-02 -> 2026-03-06)",
        body="<html><body>India report</body></html>",
    )


def _make_india_allocation_activities(
    universe_data, hrp, summary, email, *, universe_error=None
):
    """Build mock activities for IndiaWeeklyAllocationWorkflow."""

    @activity.defn(name="get_halal_india_universe")
    def mock_get_halal_india_universe() -> dict:
        if universe_error:
            raise universe_error
        return universe_data

    @activity.defn(name="allocate_hrp")
    def mock_allocate_hrp(symbols, as_of_date, lookback_days=252):
        return hrp

    @activity.defn(name="generate_india_alpha_hrp_summary")
    def mock_generate_india_alpha_hrp_summary(hrp_arg, universe):
        return summary

    @activity.defn(name="send_india_alpha_hrp_email")
    def mock_send_india_alpha_hrp_email(
        summary_arg, hrp_arg, universe, start, end, as_of
    ):
        return email

    return [
        mock_get_halal_india_universe,
        mock_allocate_hrp,
        mock_generate_india_alpha_hrp_summary,
        mock_send_india_alpha_hrp_email,
    ]


class TestIndiaWeeklyAllocationWorkflow:
    @pytest.mark.asyncio
    async def test_full_workflow_success(
        self, mock_universe_data, mock_hrp, mock_summary, mock_email
    ):
        activities = _make_india_allocation_activities(
            mock_universe_data, mock_hrp, mock_summary, mock_email
        )

        async with await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter
        ) as env:
            async with Worker(
                env.client,
                task_queue="test-queue",
                workflows=[IndiaWeeklyAllocationWorkflow],
                activities=activities,
                activity_executor=ThreadPoolExecutor(),
            ):
                result = await env.client.execute_workflow(
                    IndiaWeeklyAllocationWorkflow.run,
                    id="test-india-email",
                    task_queue="test-queue",
                )

            assert result["universe_stocks"] == 3
            assert result["hrp_symbols"] == 3
            assert result["summary_provider"] == "openai"
            assert result["email"]["is_success"] is True
            assert "India" in result["email"]["subject"]

    @pytest.mark.asyncio
    async def test_target_week_end_is_4_days_after_start(
        self, mock_universe_data, mock_hrp, mock_summary, mock_email
    ):
        activities = _make_india_allocation_activities(
            mock_universe_data, mock_hrp, mock_summary, mock_email
        )

        async with await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter
        ) as env:
            async with Worker(
                env.client,
                task_queue="test-queue",
                workflows=[IndiaWeeklyAllocationWorkflow],
                activities=activities,
                activity_executor=ThreadPoolExecutor(),
            ):
                result = await env.client.execute_workflow(
                    IndiaWeeklyAllocationWorkflow.run,
                    id="test-india-email-dates",
                    task_queue="test-queue",
                )

            start = datetime.strptime(result["target_week_start"], "%Y-%m-%d")
            end = datetime.strptime(result["target_week_end"], "%Y-%m-%d")
            assert (end - start).days == 4

    @pytest.mark.asyncio
    async def test_hrp_called_with_universe_symbols(
        self, mock_universe_data, mock_hrp, mock_summary, mock_email
    ):
        hrp_calls = []

        @activity.defn(name="get_halal_india_universe")
        def mock_universe():
            return mock_universe_data

        @activity.defn(name="allocate_hrp")
        def mock_allocate(symbols, as_of_date, lookback_days=252):
            hrp_calls.append({"symbols": symbols, "as_of_date": as_of_date})
            return mock_hrp

        @activity.defn(name="generate_india_alpha_hrp_summary")
        def mock_gen(hrp_arg, universe):
            return mock_summary

        @activity.defn(name="send_india_alpha_hrp_email")
        def mock_send(summary_arg, hrp_arg, universe, start, end, as_of):
            return mock_email

        async with await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter
        ) as env:
            async with Worker(
                env.client,
                task_queue="test-queue",
                workflows=[IndiaWeeklyAllocationWorkflow],
                activities=[mock_universe, mock_allocate, mock_gen, mock_send],
                activity_executor=ThreadPoolExecutor(),
            ):
                await env.client.execute_workflow(
                    IndiaWeeklyAllocationWorkflow.run,
                    id="test-india-email-hrp",
                    task_queue="test-queue",
                )

            assert len(hrp_calls) == 1
            expected_symbols = [s["symbol"] for s in mock_universe_data["stocks"]]
            assert hrp_calls[0]["symbols"] == expected_symbols


class TestIndiaWorkflowFailurePropagation:
    @pytest.mark.asyncio
    async def test_universe_failure_stops_workflow(
        self, mock_universe_data, mock_hrp, mock_summary, mock_email
    ):
        activities = _make_india_allocation_activities(
            mock_universe_data,
            mock_hrp,
            mock_summary,
            mock_email,
            universe_error=RuntimeError("NSE API down"),
        )

        async with (
            await WorkflowEnvironment.start_time_skipping(
                data_converter=pydantic_data_converter
            ) as env,
            Worker(
                env.client,
                task_queue="test-queue",
                workflows=[IndiaWeeklyAllocationWorkflow],
                activities=activities,
                activity_executor=ThreadPoolExecutor(),
            ),
        ):
            with pytest.raises(WorkflowFailureError):
                await env.client.execute_workflow(
                    IndiaWeeklyAllocationWorkflow.run,
                    id="test-india-email-fail",
                    task_queue="test-queue",
                )

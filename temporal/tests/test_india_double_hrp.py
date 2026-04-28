"""Tests for India Double HRP Temporal workflow.

Tests:
- Full two-stage workflow execution with mocked activities
- Stage 1 called with all symbols and 756-day lookback
- Top 15 selection from Stage 1 weights
- Stage 2 called with top 15 symbols and 252-day lookback
- Email sent with both stages
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pytest
from temporalio import activity
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from models import (
    HRPAllocationResponse,
    WeeklyReportEmailResponse,
    WeeklySummaryResponse,
)
from workflows.india_double_hrp import IndiaDoubleHRPWorkflow


@pytest.fixture
def mock_universe_data():
    """~210 Nifty Shariah 500 symbols (simulated with 20)."""
    return {
        "stocks": [{"symbol": f"SYM{i}.NS"} for i in range(20)],
        "total_stocks": 20,
        "source": "nifty_500_shariah",
    }


@pytest.fixture
def mock_stage1():
    """Stage 1: full universe HRP result (sorted descending by weight)."""
    weights = {f"SYM{i}.NS": round(20.0 - i * 0.8, 2) for i in range(20)}
    return HRPAllocationResponse(
        percentage_weights=weights,
        symbols_used=20,
        symbols_excluded=[],
        lookback_days=756,
        as_of_date="2026-04-28",
    )


@pytest.fixture
def mock_stage2():
    """Stage 2: top 15 HRP result."""
    weights = {f"SYM{i}.NS": round(100.0 / 15, 2) for i in range(15)}
    return HRPAllocationResponse(
        percentage_weights=weights,
        symbols_used=15,
        symbols_excluded=[],
        lookback_days=252,
        as_of_date="2026-04-28",
    )


@pytest.fixture
def mock_summary():
    return WeeklySummaryResponse(
        summary={
            "para_1_screening_overview": "Stage 1 distributed weights across 20 stocks."
        },
        provider="openai",
        model_used="gpt-4o-mini",
        tokens_used=400,
    )


@pytest.fixture
def mock_email():
    return WeeklyReportEmailResponse(
        is_success=True,
        subject="Double HRP Portfolio Analysis (2026-04-28 -> 2026-05-02)",
        body="<html><body>Double HRP report</body></html>",
    )


def _make_double_hrp_activities(
    universe_data,
    stage1,
    stage2,
    summary,
    email,
    *,
    hrp_calls=None,
):
    """Build mock activities for IndiaDoubleHRPWorkflow."""

    @activity.defn(name="fetch_nifty_shariah_500_universe")
    def mock_fetch_universe() -> dict:
        return universe_data

    @activity.defn(name="allocate_hrp")
    def mock_allocate_hrp(symbols, as_of_date, lookback_days=252):
        if hrp_calls is not None:
            hrp_calls.append(
                {
                    "symbols": symbols,
                    "as_of_date": as_of_date,
                    "lookback_days": lookback_days,
                }
            )
        if lookback_days == 756:
            return stage1
        return stage2

    @activity.defn(name="generate_double_hrp_summary")
    def mock_generate_summary(s1, s2, universe, top_n):
        return summary

    @activity.defn(name="send_double_hrp_email")
    def mock_send_email(summ, s1, s2, universe, top_n, start, end, as_of):
        return email

    return [
        mock_fetch_universe,
        mock_allocate_hrp,
        mock_generate_summary,
        mock_send_email,
    ]


class TestIndiaDoubleHRPWorkflow:
    @pytest.mark.asyncio
    async def test_full_workflow_success(
        self, mock_universe_data, mock_stage1, mock_stage2, mock_summary, mock_email
    ):
        activities = _make_double_hrp_activities(
            mock_universe_data, mock_stage1, mock_stage2, mock_summary, mock_email
        )

        async with await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter
        ) as env:
            async with Worker(
                env.client,
                task_queue="test-queue",
                workflows=[IndiaDoubleHRPWorkflow],
                activities=activities,
                activity_executor=ThreadPoolExecutor(),
            ):
                result = await env.client.execute_workflow(
                    IndiaDoubleHRPWorkflow.run,
                    id="test-double-hrp",
                    task_queue="test-queue",
                )

            assert result["universe_symbols"] == 20
            assert result["stage1_symbols_used"] == 20
            assert result["top_n"] == 15
            assert len(result["top_n_symbols"]) == 15
            assert result["stage2_symbols_used"] == 15
            assert result["summary_provider"] == "openai"
            assert result["email"]["is_success"] is True
            assert "Double HRP" in result["email"]["subject"]

    @pytest.mark.asyncio
    async def test_stage1_uses_756_lookback_stage2_uses_252(
        self, mock_universe_data, mock_stage1, mock_stage2, mock_summary, mock_email
    ):
        hrp_calls = []
        activities = _make_double_hrp_activities(
            mock_universe_data,
            mock_stage1,
            mock_stage2,
            mock_summary,
            mock_email,
            hrp_calls=hrp_calls,
        )

        async with await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter
        ) as env:
            async with Worker(
                env.client,
                task_queue="test-queue",
                workflows=[IndiaDoubleHRPWorkflow],
                activities=activities,
                activity_executor=ThreadPoolExecutor(),
            ):
                await env.client.execute_workflow(
                    IndiaDoubleHRPWorkflow.run,
                    id="test-double-hrp-lookbacks",
                    task_queue="test-queue",
                )

            assert len(hrp_calls) == 2

            # Stage 1: all symbols, 756-day lookback
            assert hrp_calls[0]["lookback_days"] == 756
            expected_symbols = [s["symbol"] for s in mock_universe_data["stocks"]]
            assert hrp_calls[0]["symbols"] == expected_symbols

            # Stage 2: top 15 symbols, 252-day lookback
            assert hrp_calls[1]["lookback_days"] == 252
            assert len(hrp_calls[1]["symbols"]) == 15

    @pytest.mark.asyncio
    async def test_top_15_selected_from_stage1_weights(
        self, mock_universe_data, mock_stage1, mock_stage2, mock_summary, mock_email
    ):
        hrp_calls = []
        activities = _make_double_hrp_activities(
            mock_universe_data,
            mock_stage1,
            mock_stage2,
            mock_summary,
            mock_email,
            hrp_calls=hrp_calls,
        )

        async with await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter
        ) as env:
            async with Worker(
                env.client,
                task_queue="test-queue",
                workflows=[IndiaDoubleHRPWorkflow],
                activities=activities,
                activity_executor=ThreadPoolExecutor(),
            ):
                result = await env.client.execute_workflow(
                    IndiaDoubleHRPWorkflow.run,
                    id="test-double-hrp-selection",
                    task_queue="test-queue",
                )

            # Top 15 should be the first 15 keys from stage1 (sorted descending)
            expected_top_15 = list(mock_stage1.percentage_weights.keys())[:15]
            assert result["top_n_symbols"] == expected_top_15
            assert hrp_calls[1]["symbols"] == expected_top_15

    @pytest.mark.asyncio
    async def test_target_week_end_is_4_days_after_start(
        self, mock_universe_data, mock_stage1, mock_stage2, mock_summary, mock_email
    ):
        activities = _make_double_hrp_activities(
            mock_universe_data, mock_stage1, mock_stage2, mock_summary, mock_email
        )

        async with await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter
        ) as env:
            async with Worker(
                env.client,
                task_queue="test-queue",
                workflows=[IndiaDoubleHRPWorkflow],
                activities=activities,
                activity_executor=ThreadPoolExecutor(),
            ):
                result = await env.client.execute_workflow(
                    IndiaDoubleHRPWorkflow.run,
                    id="test-double-hrp-dates",
                    task_queue="test-queue",
                )

            start = datetime.strptime(result["target_week_start"], "%Y-%m-%d")
            end = datetime.strptime(result["target_week_end"], "%Y-%m-%d")
            assert (end - start).days == 4

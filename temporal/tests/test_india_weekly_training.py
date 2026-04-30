"""Tests for India weekly training Temporal workflow."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from temporalio import activity
from temporalio.client import WorkflowFailureError
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from models import (
    TrainingResponse,
    TrainingSummaryEmailResponse,
    TrainingSummaryResponse,
)
from workflows.india_weekly_training import IndiaWeeklyTrainingWorkflow


@pytest.fixture
def mock_nifty():
    return {
        "stocks": [{"symbol": f"SYM{i}.NS"} for i in range(210)],
        "total_stocks": 210,
    }


@pytest.fixture
def mock_training():
    return TrainingResponse(
        version="v2026-03-01-india123",
        data_window_start="2015-01-01",
        data_window_end="2025-12-26",
        metrics={"train_loss": 0.01, "val_loss": 0.02},
        promoted=True,
        num_input_channels=5,
        signals_used=["ohlcv"],
    )


@pytest.fixture
def mock_india_filtered():
    return {
        "stocks": [
            {
                "symbol": f"TOP{i}.NS",
                "predicted_weekly_return_pct": 5.0 - i * 0.3,
                "rank": i + 1,
                "selection_reason": "top_rank",
            }
            for i in range(15)
        ],
        "total_candidates": 180,
        "total_universe": 210,
        "filtered_insufficient_history": 30,
        "top_n": 15,
        "selection_method": "patchtst_forecast_rank_band",
        "model_version": "v2026-03-01-india123",
        "symbol_suffix": ".NS",
        "fetched_at": "2026-04-01T00:00:00+00:00",
        "partition": "halal_india_filtered_alpha",
        "period_key": "202615",
        "previous_period_key_used": None,
        "kept_count": 0,
        "fillers_count": 15,
        "evicted_from_previous": {},
        "k_in": 15,
        "k_hold": 30,
    }


@pytest.fixture
def mock_summary():
    return TrainingSummaryResponse(
        summary={"para_1_overall": "India PatchTST training completed successfully."},
        provider="openai",
        model_used="gpt-4o-mini",
        tokens_used=300,
    )


@pytest.fixture
def mock_email():
    return TrainingSummaryEmailResponse(
        is_success=True,
        subject="India Training Summary: 2015-01-01 to 2025-12-26",
        body="<html><body>India training summary</body></html>",
    )


def _make_india_training_activities(
    nifty, training, india_filtered, summary, email, *, nifty_error=None
):
    @activity.defn(name="fetch_nifty_shariah_500_universe")
    def mock_nifty_fn():
        if nifty_error:
            raise nifty_error
        return nifty

    @activity.defn(name="train_india_patchtst")
    def mock_train():
        return training

    @activity.defn(name="fetch_halal_india_universe")
    def mock_filtered():
        return india_filtered

    @activity.defn(name="generate_india_training_summary")
    def mock_summary_fn(patchtst):
        return summary

    @activity.defn(name="send_india_training_email")
    def mock_email_fn(patchtst, summary_arg):
        return email

    return [mock_nifty_fn, mock_train, mock_filtered, mock_summary_fn, mock_email_fn]


class TestIndiaWeeklyTrainingWorkflow:
    @pytest.mark.asyncio
    async def test_full_workflow_success(
        self, mock_nifty, mock_training, mock_india_filtered, mock_summary, mock_email
    ):
        activities = _make_india_training_activities(
            mock_nifty, mock_training, mock_india_filtered, mock_summary, mock_email
        )

        async with await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter
        ) as env:
            async with Worker(
                env.client,
                task_queue="test-queue",
                workflows=[IndiaWeeklyTrainingWorkflow],
                activities=activities,
                activity_executor=ThreadPoolExecutor(),
            ):
                result = await env.client.execute_workflow(
                    IndiaWeeklyTrainingWorkflow.run,
                    id="test-india-training",
                    task_queue="test-queue",
                )

            assert result["nifty_shariah_500"]["total_stocks"] == 210
            assert result["patchtst"]["version"] == "v2026-03-01-india123"
            assert result["patchtst"]["promoted"] is True
            assert result["halal_india"]["stocks"] == 15
            assert (
                result["halal_india"]["selection_method"]
                == "patchtst_forecast_rank_band"
            )
            assert result["summary"]["provider"] == "openai"
            assert result["email"]["is_success"] is True
            assert "India" in result["email"]["subject"]


class TestIndiaTrainingFailures:
    @pytest.mark.asyncio
    async def test_universe_failure_stops_workflow(
        self, mock_nifty, mock_training, mock_india_filtered, mock_summary, mock_email
    ):
        activities = _make_india_training_activities(
            mock_nifty,
            mock_training,
            mock_india_filtered,
            mock_summary,
            mock_email,
            nifty_error=RuntimeError("NSE API down"),
        )

        async with (
            await WorkflowEnvironment.start_time_skipping(
                data_converter=pydantic_data_converter
            ) as env,
            Worker(
                env.client,
                task_queue="test-queue",
                workflows=[IndiaWeeklyTrainingWorkflow],
                activities=activities,
                activity_executor=ThreadPoolExecutor(),
            ),
        ):
            with pytest.raises(WorkflowFailureError):
                await env.client.execute_workflow(
                    IndiaWeeklyTrainingWorkflow.run,
                    id="test-india-training-fail",
                    task_queue="test-queue",
                )

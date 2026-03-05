"""Tests for US weekly training Temporal workflow."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from temporalio import activity
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from models import (
    RefreshTrainingDataResponse,
    TrainingResponse,
    TrainingSummaryEmailResponse,
    TrainingSummaryResponse,
)
from workflows.us_weekly_training import USWeeklyTrainingWorkflow


@pytest.fixture
def mock_halal_new():
    return {"stocks": [{"symbol": f"SYM{i}"} for i in range(410)], "total_stocks": 410}


@pytest.fixture
def mock_filtered():
    return {
        "stocks": [{"symbol": f"TOP{i}", "rank": i + 1} for i in range(15)],
        "total_candidates": 380,
        "top_n": 15,
        "selection_method": "patchtst_forecast",
        "model_version": "v2026-03-01-abc123",
    }


@pytest.fixture
def mock_training():
    return TrainingResponse(
        version="v1.0.0",
        data_window_start="2020-01-01",
        data_window_end="2024-01-01",
        metrics={"loss": 0.01},
        promoted=True,
    )


@pytest.fixture
def mock_refresh():
    return RefreshTrainingDataResponse(
        sentiment_gaps_filled=10,
        sentiment_gaps_remaining=0,
        fundamentals_refreshed=["AAPL", "MSFT"],
        fundamentals_skipped=["GOOGL"],
        fundamentals_failed=[],
        duration_seconds=5.5,
    )


@pytest.fixture
def mock_summary():
    return TrainingSummaryResponse(
        summary={"para_1_overall": "All models trained successfully."},
        provider="openai",
        model_used="gpt-4o-mini",
        tokens_used=500,
    )


@pytest.fixture
def mock_email():
    return TrainingSummaryEmailResponse(
        is_success=True,
        subject="Training Summary: 2020-01-01 to 2024-01-01",
        body="<html><body>Training summary</body></html>",
    )


def _make_us_training_activities(
    halal_new, filtered, training, refresh, summary, email
):
    @activity.defn(name="fetch_halal_new_universe")
    def mock_new():
        return halal_new

    @activity.defn(name="train_lstm")
    def mock_lstm():
        return training

    @activity.defn(name="train_patchtst")
    def mock_ptst():
        return training

    @activity.defn(name="fetch_halal_filtered_universe")
    def mock_filt():
        return filtered

    @activity.defn(name="refresh_training_data")
    def mock_ref():
        return refresh

    @activity.defn(name="train_ppo")
    def mock_ppo():
        return training

    @activity.defn(name="train_sac")
    def mock_sac():
        return training

    @activity.defn(name="generate_training_summary")
    def mock_summ(lstm, patchtst, ppo, sac):
        return summary

    @activity.defn(name="send_training_summary_email")
    def mock_em(lstm, patchtst, ppo, sac, summary_arg):
        return email

    return [
        mock_new,
        mock_lstm,
        mock_ptst,
        mock_filt,
        mock_ref,
        mock_ppo,
        mock_sac,
        mock_summ,
        mock_em,
    ]


class TestUSWeeklyTrainingWorkflow:
    @pytest.mark.asyncio
    async def test_full_workflow_success(
        self,
        mock_halal_new,
        mock_filtered,
        mock_training,
        mock_refresh,
        mock_summary,
        mock_email,
    ):
        activities = _make_us_training_activities(
            mock_halal_new,
            mock_filtered,
            mock_training,
            mock_refresh,
            mock_summary,
            mock_email,
        )

        async with await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter
        ) as env:
            async with Worker(
                env.client,
                task_queue="test-queue",
                workflows=[USWeeklyTrainingWorkflow],
                activities=activities,
                activity_executor=ThreadPoolExecutor(),
            ):
                result = await env.client.execute_workflow(
                    USWeeklyTrainingWorkflow.run,
                    id="test-us-training",
                    task_queue="test-queue",
                )

            assert result["halal_new"]["total_stocks"] == 410
            assert result["lstm"]["version"] == "v1.0.0"
            assert result["lstm"]["promoted"] is True
            assert result["patchtst"]["version"] == "v1.0.0"
            assert result["filtered"]["stocks"] == 15
            assert result["filtered"]["selection_method"] == "patchtst_forecast"
            assert result["refresh"]["sentiment_gaps_filled"] == 10
            assert result["refresh"]["fundamentals_refreshed"] == 2
            assert result["ppo"]["version"] == "v1.0.0"
            assert result["sac"]["version"] == "v1.0.0"
            assert result["summary"]["provider"] == "openai"
            assert result["email"]["is_success"] is True
            assert "Training Summary" in result["email"]["subject"]

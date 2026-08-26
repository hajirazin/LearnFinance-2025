"""Tests for the US Forecasters training Temporal workflow.

Mirrors the structure of the (deleted) ``test_us_weekly_training.py``
suite but focuses purely on the LSTM + PatchTST half of the split. SAC
lives in :mod:`temporal/tests/test_us_sac_training.py`.
"""

from concurrent.futures import ThreadPoolExecutor

import pytest
from temporalio import activity
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from models import (
    TrainingResponse,
    TrainingSummaryEmailResponse,
    TrainingSummaryResponse,
)
from workflows.us_forecasters_training import USForecastersTrainingWorkflow


@pytest.fixture
def mock_halal_new():
    return {"stocks": [{"symbol": f"SYM{i}"} for i in range(410)], "total_stocks": 410}


@pytest.fixture
def mock_training():
    return TrainingResponse(
        version="v1.0.0",
        data_window_start="2020-01-01",
        data_window_end="2024-01-01",
        metrics={"loss": 0.01},
        promoted=True,
        failure_reasons=[],
    )


@pytest.fixture
def mock_summary():
    return TrainingSummaryResponse(
        summary={"para_1_overall": "Forecasters trained successfully."},
        provider="openai",
        model_used="gpt-5-mini",
        tokens_used=300,
    )


@pytest.fixture
def mock_email():
    return TrainingSummaryEmailResponse(
        is_success=True,
        subject="US Forecasters Training: 2020-01-01 to 2024-01-01",
        body="<html><body>Forecasters summary</body></html>",
    )


def _make_forecasters_activities(halal_new, training, summary, email):
    """Build mocked activities that match the registered names exactly.

    The workflow looks up activities by name (defined via
    ``@activity.defn``), so each mock must use the same name as the
    real activity in :mod:`activities.training`.
    """
    call_log: list[str] = []

    @activity.defn(name="fetch_halal_new_universe")
    def mock_new():
        call_log.append("fetch_halal_new_universe")
        return halal_new

    @activity.defn(name="train_lstm")
    def mock_lstm(universe: str):
        call_log.append("train_lstm")
        assert universe == "halal_new"
        return training

    @activity.defn(name="train_patchtst")
    def mock_ptst(universe: str):
        call_log.append("train_patchtst")
        assert universe == "halal_new"
        return training

    @activity.defn(name="generate_forecasters_training_summary")
    def mock_summ(lstm, patchtst):
        call_log.append("generate_forecasters_training_summary")
        return summary

    @activity.defn(name="send_forecasters_training_email")
    def mock_em(lstm, patchtst, summary_arg):
        call_log.append("send_forecasters_training_email")
        return email

    return [mock_new, mock_lstm, mock_ptst, mock_summ, mock_em], call_log


class TestUSForecastersTrainingWorkflow:
    @pytest.mark.asyncio
    async def test_full_workflow_success(
        self,
        mock_halal_new,
        mock_training,
        mock_summary,
        mock_email,
    ):
        activities, call_log = _make_forecasters_activities(
            mock_halal_new,
            mock_training,
            mock_summary,
            mock_email,
        )

        async with await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter
        ) as env:
            async with Worker(
                env.client,
                task_queue="test-queue",
                workflows=[USForecastersTrainingWorkflow],
                activities=activities,
                activity_executor=ThreadPoolExecutor(),
            ):
                result = await env.client.execute_workflow(
                    USForecastersTrainingWorkflow.run,
                    id="test-us-forecasters-training",
                    task_queue="test-queue",
                )

            assert result["halal_new"]["total_stocks"] == 410
            assert result["lstm"]["version"] == "v1.0.0"
            assert result["lstm"]["promoted"] is True
            assert result["lstm"]["failure_reasons"] == []
            assert result["patchtst"]["version"] == "v1.0.0"
            assert result["patchtst"]["promoted"] is True
            assert result["patchtst"]["failure_reasons"] == []
            assert result["summary"]["provider"] == "openai"
            assert result["email"]["is_success"] is True
            assert "US Forecasters Training" in result["email"]["subject"]

            # The two training activities must run strictly serially
            # (LSTM before PatchTST) because the host cannot fit both
            # concurrently. Verify by call order.
            lstm_idx = call_log.index("train_lstm")
            ptst_idx = call_log.index("train_patchtst")
            assert lstm_idx < ptst_idx

            # No SAC, refresh, or filtered-fetch activities should be
            # invoked from the forecasters workflow.
            assert "train_sac" not in call_log
            assert "run_news_backfill" not in call_log
            assert "fetch_halal_filtered_universe" not in call_log

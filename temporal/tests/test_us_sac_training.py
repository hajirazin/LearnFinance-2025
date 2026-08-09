"""Tests for the US SAC training Temporal workflow.

The SAC workflow is the second half of the (deleted)
``USWeeklyTrainingWorkflow`` split. It runs a day after the forecasters
workflow and consumes whatever PatchTST ``current`` pointer is live at
trigger time, so it does NOT trigger forecaster training itself.
"""

from concurrent.futures import ThreadPoolExecutor

import pytest
from temporalio import activity
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from models import (
    SACReadinessIssue,
    SACTrainingReadiness,
    SACTrainingWorkflowInput,
    SentimentGapFillResponse,
    TrainingResponse,
    TrainingSummaryEmailResponse,
    TrainingSummaryResponse,
)
from workflows.us_sac_training import USSACTrainingWorkflow


@pytest.fixture
def mock_filtered():
    return {
        "stocks": [{"symbol": f"TOP{i}", "rank": i + 1} for i in range(15)],
        "total_candidates": 380,
        "top_n": 15,
        "selection_method": "patchtst_forecast_rank_band",
        "model_version": "v2026-03-01-abc123",
    }


@pytest.fixture
def mock_refresh():
    return SentimentGapFillResponse(
        rows_added=10,
        remaining_gaps=0,
        gaps_pre_api_date=3,
        duration_seconds=5.5,
        hf_url="https://huggingface.co/datasets/example/news",
    )


@pytest.fixture
def mock_sac_training():
    return TrainingResponse(
        version="v2026-03-01-sac001",
        data_window_start="2020-01-01",
        data_window_end="2024-01-01",
        metrics={"sharpe": 1.5, "cagr": 0.18},
        promoted=True,
        failure_reasons=[],
        symbols_used=["AAPL", "MSFT", "GOOGL"],
    )


@pytest.fixture
def mock_summary():
    return TrainingSummaryResponse(
        summary={"para_1_overall": "SAC trained successfully."},
        provider="openai",
        model_used="gpt-5-mini",
        tokens_used=200,
    )


@pytest.fixture
def mock_email():
    return TrainingSummaryEmailResponse(
        is_success=True,
        subject="US SAC Training: 2020-01-01 to 2024-01-01",
        body="<html><body>SAC summary</body></html>",
    )


def _make_sac_activities(
    filtered,
    refresh,
    training,
    summary,
    email,
    *,
    expected_force: bool = False,
):
    """Build mocked activities matching the registered names exactly."""
    call_log: list[str] = []
    preflight_calls = 0

    @activity.defn(name="fetch_halal_filtered_universe")
    def mock_filt():
        call_log.append("fetch_halal_filtered_universe")
        return filtered

    @activity.defn(name="run_sentiment_gap_fill")
    def mock_ref(universe: str):
        call_log.append("run_sentiment_gap_fill")
        # SAC workflow refreshes the same slate it trains on.
        assert universe == "halal_filtered"
        return refresh

    @activity.defn(name="preflight_sac_training")
    def mock_preflight(universe: str, force: bool = False):
        nonlocal preflight_calls
        call_log.append("preflight_sac_training")
        assert universe == "halal_filtered"
        assert force is expected_force
        preflight_calls += 1
        if preflight_calls == 1:
            return SACTrainingReadiness(
                universe=universe,
                symbols=[stock["symbol"] for stock in filtered["stocks"]],
                ready=False,
                missing=[
                    SACReadinessIssue(
                        source="news",
                        detail="quota refresh required",
                        retryable=True,
                    )
                ],
            )
        return SACTrainingReadiness(
            universe=universe,
            symbols=[stock["symbol"] for stock in filtered["stocks"]],
            ready=True,
        )

    @activity.defn(name="train_sac")
    def mock_sac(universe: str, force: bool = False):
        call_log.append("train_sac")
        assert universe == "halal_filtered"
        assert force is expected_force
        return training

    @activity.defn(name="generate_sac_training_summary")
    def mock_summ(sac, universe: str):
        call_log.append("generate_sac_training_summary")
        assert universe == "halal_filtered", (
            f"expected summary universe=halal_filtered, got {universe!r}"
        )
        return summary

    @activity.defn(name="send_sac_training_email")
    def mock_em(sac, summary_arg, universe: str):
        call_log.append("send_sac_training_email")
        assert universe == "halal_filtered", (
            f"expected email universe=halal_filtered, got {universe!r}"
        )
        return email

    return [
        mock_filt,
        mock_preflight,
        mock_ref,
        mock_sac,
        mock_summ,
        mock_em,
    ], call_log


class TestUSSACTrainingWorkflow:
    @pytest.mark.asyncio
    async def test_full_workflow_success(
        self,
        mock_filtered,
        mock_refresh,
        mock_sac_training,
        mock_summary,
        mock_email,
    ):
        activities, call_log = _make_sac_activities(
            mock_filtered,
            mock_refresh,
            mock_sac_training,
            mock_summary,
            mock_email,
        )

        async with await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter
        ) as env:
            async with Worker(
                env.client,
                task_queue="test-queue",
                workflows=[USSACTrainingWorkflow],
                activities=activities,
                activity_executor=ThreadPoolExecutor(),
            ):
                result = await env.client.execute_workflow(
                    USSACTrainingWorkflow.run,
                    id="test-us-sac-training",
                    task_queue="test-queue",
                )

            assert result["filtered"]["stocks"] == 15
            assert result["filtered"]["selection_method"] == (
                "patchtst_forecast_rank_band"
            )
            assert result["refresh"]["rows_added"] == 10
            assert result["refresh"]["gaps_pre_api_date"] == 3
            assert result["refresh"]["published"] is True
            assert result["sac"]["version"] == "v2026-03-01-sac001"
            assert result["sac"]["promoted"] is True
            assert result["sac"]["failure_reasons"] == []
            assert result["readiness"] == {"ready": True, "attempts": 2}
            assert result["summary"]["provider"] == "openai"
            assert result["email"]["is_success"] is True
            assert "US SAC Training" in result["email"]["subject"]

            # SAC must not retrain forecasters or rebuild halal_new.
            assert "train_lstm" not in call_log
            assert "train_patchtst" not in call_log
            assert "fetch_halal_new_universe" not in call_log

            # filtered-fetch must precede refresh + SAC train.
            filt_idx = call_log.index("fetch_halal_filtered_universe")
            preflight_idx = call_log.index("preflight_sac_training")
            ref_idx = call_log.index("run_sentiment_gap_fill")
            sac_idx = call_log.index("train_sac")
            assert filt_idx < preflight_idx < ref_idx < sac_idx

    @pytest.mark.asyncio
    async def test_manual_force_is_forwarded_to_training_activity(
        self,
        mock_filtered,
        mock_refresh,
        mock_sac_training,
        mock_summary,
        mock_email,
    ):
        activities, _ = _make_sac_activities(
            mock_filtered,
            mock_refresh,
            mock_sac_training,
            mock_summary,
            mock_email,
            expected_force=True,
        )

        async with (
            await WorkflowEnvironment.start_time_skipping(
                data_converter=pydantic_data_converter
            ) as env,
            Worker(
                env.client,
                task_queue="test-queue",
                workflows=[USSACTrainingWorkflow],
                activities=activities,
                activity_executor=ThreadPoolExecutor(),
            ),
        ):
            result = await env.client.execute_workflow(
                USSACTrainingWorkflow.run,
                SACTrainingWorkflowInput(force=True),
                id="test-us-sac-training-force",
                task_queue="test-queue",
            )

        assert result["sac"]["force"] is True

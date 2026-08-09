"""Tests for the US SAC (halal) training Temporal workflow.

Sibling of :mod:`tests.test_us_sac_training`. Both workflows train
SAC on a Sunday cron slot but on different universes for an A/B
comparison; this one uses the legacy ``halal`` universe (yfinance
ETF top-holdings, variable size). Per the registry contract, the
workflow MUST:

- Fetch the ``halal`` universe (NOT ``halal_filtered`` or ``halal_new``).
- Call ``run_sentiment_gap_fill`` and ``train_sac`` with
  ``universe="halal"``.
- Forward ``universe="halal"`` to the summary + email activities so
  downstream brain_api endpoints can branch the prompt and subject
  on the bucket.
- NOT trigger forecaster training or fetch any other universe.
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
    SentimentGapFillResponse,
    TrainingResponse,
    TrainingSummaryEmailResponse,
    TrainingSummaryResponse,
)
from workflows.us_sac_halal_training import USSACHalalTrainingWorkflow


@pytest.fixture
def mock_halal():
    """yfinance halal universe is variable-size; emulate with 14 names."""
    stocks = [{"symbol": f"HAL{i}"} for i in range(14)]
    return {
        "stocks": stocks,
        "total_stocks": len(stocks),
    }


@pytest.fixture
def mock_refresh():
    return SentimentGapFillResponse(
        rows_added=7,
        remaining_gaps=0,
        gaps_pre_api_date=2,
        duration_seconds=4.2,
        hf_url="https://huggingface.co/datasets/example/news",
    )


@pytest.fixture
def mock_sac_training():
    return TrainingResponse(
        version="v2026-03-01-sac-halal",
        data_window_start="2020-01-01",
        data_window_end="2024-01-01",
        metrics={"sharpe": 1.4, "cagr": 0.16},
        promoted=True,
        failure_reasons=[],
        symbols_used=[f"HAL{i}" for i in range(14)],
    )


@pytest.fixture
def mock_summary():
    return TrainingSummaryResponse(
        summary={"para_1_overall": "SAC (halal) trained successfully."},
        provider="openai",
        model_used="gpt-5-mini",
        tokens_used=180,
    )


@pytest.fixture
def mock_email():
    return TrainingSummaryEmailResponse(
        is_success=True,
        subject="US SAC (halal) Training: 2020-01-01 to 2024-01-01",
        body="<html><body>SAC (halal) summary</body></html>",
    )


def _make_sac_activities(halal, refresh, training, summary, email):
    """Build mocked activities matching the registered names exactly.

    Each mock asserts that the universe argument it receives matches
    the bucket the halal workflow is supposed to drive. If a future
    refactor accidentally rewires this workflow to ``halal_filtered``
    (or any other universe), these assertions catch it before the
    bucket buckets clobber each other on disk.
    """
    call_log: list[str] = []
    preflight_calls = 0

    @activity.defn(name="fetch_halal_universe")
    def mock_halal_fetch():
        call_log.append("fetch_halal_universe")
        return halal

    @activity.defn(name="run_sentiment_gap_fill")
    def mock_ref(universe: str):
        call_log.append("run_sentiment_gap_fill")
        assert universe == "halal", (
            f"expected refresh on universe=halal, got {universe!r}"
        )
        return refresh

    @activity.defn(name="preflight_sac_training")
    def mock_preflight(universe: str, force: bool = False):
        nonlocal preflight_calls
        call_log.append("preflight_sac_training")
        assert universe == "halal"
        assert force is False
        preflight_calls += 1
        if preflight_calls == 1:
            return SACTrainingReadiness(
                universe=universe,
                symbols=[stock["symbol"] for stock in halal["stocks"]],
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
            symbols=[stock["symbol"] for stock in halal["stocks"]],
            ready=True,
        )

    @activity.defn(name="train_sac")
    def mock_sac(universe: str, force: bool = False):
        call_log.append("train_sac")
        assert universe == "halal", (
            f"expected train_sac universe=halal, got {universe!r}"
        )
        assert force is False
        return training

    @activity.defn(name="generate_sac_training_summary")
    def mock_summ(sac, universe: str = "halal_filtered"):
        call_log.append("generate_sac_training_summary")
        assert universe == "halal", f"expected summary universe=halal, got {universe!r}"
        return summary

    @activity.defn(name="send_sac_training_email")
    def mock_em(sac, summary_arg, universe: str = "halal_filtered"):
        call_log.append("send_sac_training_email")
        assert universe == "halal", f"expected email universe=halal, got {universe!r}"
        return email

    return [
        mock_halal_fetch,
        mock_preflight,
        mock_ref,
        mock_sac,
        mock_summ,
        mock_em,
    ], call_log


class TestUSSACHalalTrainingWorkflow:
    @pytest.mark.asyncio
    async def test_full_workflow_success(
        self,
        mock_halal,
        mock_refresh,
        mock_sac_training,
        mock_summary,
        mock_email,
    ):
        activities, call_log = _make_sac_activities(
            mock_halal,
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
                workflows=[USSACHalalTrainingWorkflow],
                activities=activities,
                activity_executor=ThreadPoolExecutor(),
            ):
                result = await env.client.execute_workflow(
                    USSACHalalTrainingWorkflow.run,
                    id="test-us-sac-halal-training",
                    task_queue="test-queue",
                )

            assert result["halal"]["stocks"] == 14
            assert result["halal"]["total_stocks"] == 14
            assert result["refresh"]["rows_added"] == 7
            assert result["refresh"]["gaps_pre_api_date"] == 2
            assert result["refresh"]["published"] is True
            assert result["sac"]["version"] == "v2026-03-01-sac-halal"
            assert result["sac"]["promoted"] is True
            assert result["sac"]["failure_reasons"] == []
            assert result["readiness"] == {"ready": True, "attempts": 2}
            assert result["summary"]["provider"] == "openai"
            assert result["email"]["is_success"] is True
            assert "US SAC (halal) Training" in result["email"]["subject"]

            # The halal SAC workflow must NOT touch the filtered slate
            # or the forecasters; both buckets share a host so any
            # accidental cross-call would clobber the other's run.
            assert "fetch_halal_filtered_universe" not in call_log
            assert "fetch_halal_new_universe" not in call_log
            assert "fetch_nifty_shariah_500_universe" not in call_log
            assert "train_lstm" not in call_log
            assert "train_patchtst" not in call_log

            # halal-fetch must precede refresh + SAC train.
            halal_idx = call_log.index("fetch_halal_universe")
            preflight_idx = call_log.index("preflight_sac_training")
            ref_idx = call_log.index("run_sentiment_gap_fill")
            sac_idx = call_log.index("train_sac")
            assert halal_idx < preflight_idx < ref_idx < sac_idx

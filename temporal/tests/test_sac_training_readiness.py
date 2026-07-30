"""Tests for the durable SAC training-readiness gate."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from temporalio import activity, workflow
from temporalio.client import WorkflowFailureError
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from activities import training as training_module
from models import (
    RefreshTrainingDataResponse,
    SACReadinessIssue,
    SACTrainingReadiness,
)
from tests._fake_client import FakeClient
from workflows._sac_training_readiness import await_sac_training_readiness


def test_preflight_activity_forwards_universe_and_force():
    fake = FakeClient(
        {
            "/train/sac/preflight": {
                "universe": "halal",
                "symbols": ["AAPL"],
                "ready": True,
                "missing": [],
                "errors": [],
            }
        }
    )
    original = training_module.get_training_client
    training_module.get_training_client = lambda: fake
    try:
        result = training_module.preflight_sac_training("halal", force=True)
    finally:
        training_module.get_training_client = original

    assert result.ready is True
    assert fake.calls[0]["json"] == {"universe": "halal", "force": True}


@workflow.defn(sandboxed=False)
class _ReadinessDeadlineWorkflow:
    @workflow.run
    async def run(self):
        return await await_sac_training_readiness("halal_filtered", force=False)


@pytest.mark.asyncio
async def test_readiness_deadline_surfaces_exact_issues_after_seven_days():
    calls = {"preflight": 0, "refresh": 0}

    @activity.defn(name="preflight_sac_training")
    def mock_preflight(universe: str, force: bool = False):
        calls["preflight"] += 1
        return SACTrainingReadiness(
            universe=universe,
            symbols=["AAPL"],
            ready=False,
            missing=[
                SACReadinessIssue(
                    source="fundamentals",
                    symbol="AAPL",
                    detail="SEC filing availability unresolved",
                    retryable=True,
                )
            ],
        )

    @activity.defn(name="refresh_training_data")
    def mock_refresh(universe: str):
        calls["refresh"] += 1
        return RefreshTrainingDataResponse(
            sentiment_gaps_filled=0,
            sentiment_gaps_remaining=1,
            fundamentals_refreshed=[],
            fundamentals_skipped=[],
            fundamentals_failed=["AAPL"],
            duration_seconds=1.0,
        )

    async with (
        await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter
        ) as env,
        Worker(
            env.client,
            task_queue="test-readiness",
            workflows=[_ReadinessDeadlineWorkflow],
            activities=[mock_preflight, mock_refresh],
            activity_executor=ThreadPoolExecutor(),
        ),
    ):
        with pytest.raises(WorkflowFailureError) as exc_info:
            await env.client.execute_workflow(
                _ReadinessDeadlineWorkflow.run,
                id="test-readiness-deadline",
                task_queue="test-readiness",
            )

    assert calls == {"preflight": 7, "refresh": 6}
    assert "SEC filing availability unresolved" in str(exc_info.value.cause)

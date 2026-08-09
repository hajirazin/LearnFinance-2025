"""Tests for the durable SAC training-readiness gate."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from temporalio import activity, workflow
from temporalio.client import WorkflowFailureError
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.exceptions import ApplicationError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from activities import training as training_module
from models import (
    SACReadinessIssue,
    SACTrainingReadiness,
    SentimentGapFillResponse,
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


def test_sentiment_gap_activity_polls_and_requires_published_result(monkeypatch):
    fake = FakeClient(
        {
            "/etl/sentiment-gaps": {"job_id": "gap-1", "status": "pending"},
            "/etl/sentiment-gaps/gap-1": {
                "job_id": "gap-1",
                "status": "completed",
                "result": {
                    "hf_url": "https://huggingface.co/datasets/example/news",
                    "duration_seconds": 12.5,
                    "progress": {
                        "rows_added": 4,
                        "remaining_gaps": 2,
                        "gaps_pre_api_date": 11,
                    },
                },
            },
        }
    )
    heartbeats = []
    monkeypatch.setattr(training_module.activity, "heartbeat", heartbeats.append)
    monkeypatch.setattr(training_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(training_module, "get_training_client", lambda: fake)

    result = training_module.run_sentiment_gap_fill("halal", poll_interval=60.0)

    assert fake.calls[0] == {
        "method": "POST",
        "path": "/etl/sentiment-gaps",
        "json": {"universe": "halal"},
    }
    assert heartbeats == ["gap-1"]
    assert result.rows_added == 4
    assert result.gaps_pre_api_date == 11
    assert result.published is True


def test_sentiment_gap_activity_retries_completed_job_without_hf_url(monkeypatch):
    fake = FakeClient(
        {
            "/etl/sentiment-gaps": {"job_id": "gap-2", "status": "pending"},
            "/etl/sentiment-gaps/gap-2": {
                "job_id": "gap-2",
                "status": "completed",
                "result": {"hf_url": None, "progress": {}},
            },
        }
    )
    monkeypatch.setattr(training_module.activity, "heartbeat", lambda _job_id: None)
    monkeypatch.setattr(training_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(training_module, "get_training_client", lambda: fake)

    with pytest.raises(ApplicationError, match="without HF publication") as exc_info:
        training_module.run_sentiment_gap_fill("halal", poll_interval=60.0)

    assert exc_info.value.non_retryable is False


@pytest.mark.parametrize(
    ("status_code", "job_status", "message"),
    [
        (404, "running", "was lost"),
        (200, "failed", "failed: provider denied"),
    ],
)
def test_sentiment_gap_activity_surfaces_retryable_lost_or_failed_jobs(
    monkeypatch, status_code, job_status, message
):
    job_id = "gap-failed"
    status_path = f"/etl/sentiment-gaps/{job_id}"
    fake = FakeClient(
        {
            "/etl/sentiment-gaps": {"job_id": job_id, "status": "pending"},
            status_path: {
                "job_id": job_id,
                "status": job_status,
                "error": "provider denied",
            },
        },
        statuses={status_path: status_code},
    )
    monkeypatch.setattr(training_module.activity, "heartbeat", lambda _job_id: None)
    monkeypatch.setattr(training_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(training_module, "get_training_client", lambda: fake)

    with pytest.raises(ApplicationError, match=message) as exc_info:
        training_module.run_sentiment_gap_fill("halal", poll_interval=60.0)

    assert exc_info.value.non_retryable is False


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
                    source="news",
                    symbol="AAPL",
                    detail="provider observations incomplete",
                    retryable=True,
                )
            ],
        )

    @activity.defn(name="run_sentiment_gap_fill")
    def mock_refresh(universe: str):
        calls["refresh"] += 1
        return SentimentGapFillResponse(
            rows_added=0,
            remaining_gaps=1,
            gaps_pre_api_date=0,
            duration_seconds=1.0,
            hf_url="https://huggingface.co/datasets/example/news",
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
    assert "provider observations incomplete" in str(exc_info.value.cause)


@workflow.defn(sandboxed=False)
class _NonRetryableReadinessWorkflow:
    @workflow.run
    async def run(self):
        return await await_sac_training_readiness("halal_filtered", force=True)


@pytest.mark.asyncio
async def test_non_retryable_readiness_error_fails_without_refresh_or_sleep():
    calls = {"preflight": 0, "refresh": 0}

    @activity.defn(name="preflight_sac_training")
    def mock_preflight(universe: str, force: bool = False):
        calls["preflight"] += 1
        return SACTrainingReadiness(
            universe=universe,
            symbols=["AAPL"],
            ready=False,
            errors=[
                SACReadinessIssue(
                    source="news",
                    symbol="AAPL",
                    detail="Malformed news parquet",
                    retryable=False,
                )
            ],
        )

    @activity.defn(name="run_sentiment_gap_fill")
    def mock_refresh(universe: str):
        calls["refresh"] += 1
        raise AssertionError("non-retryable readiness must not refresh")

    async with (
        await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter
        ) as env,
        Worker(
            env.client,
            task_queue="test-non-retryable-readiness",
            workflows=[_NonRetryableReadinessWorkflow],
            activities=[mock_preflight, mock_refresh],
            activity_executor=ThreadPoolExecutor(),
        ),
    ):
        with pytest.raises(WorkflowFailureError) as exc_info:
            await env.client.execute_workflow(
                _NonRetryableReadinessWorkflow.run,
                id="test-non-retryable-readiness",
                task_queue="test-non-retryable-readiness",
            )

    assert calls == {"preflight": 1, "refresh": 0}
    assert "Malformed news parquet" in str(exc_info.value.cause)

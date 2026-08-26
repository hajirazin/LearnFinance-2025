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
    NewsBackfillResponse,
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


def test_news_backfill_activity_polls_until_complete(monkeypatch):
    fake = FakeClient(
        {
            "/etl/news/backfill": {"job_id": "news-1", "status": "pending"},
            "/etl/news/backfill/news-1": {
                "job_id": "news-1",
                "status": "complete",
                "windows_done": 4,
                "windows_total": 4,
                "events_scored": 12,
            },
        }
    )
    heartbeats = []
    monkeypatch.setattr(training_module.activity, "heartbeat", heartbeats.append)
    monkeypatch.setattr(training_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(training_module, "get_training_client", lambda: fake)

    result = training_module.run_news_backfill(
        ["AAPL"],
        "2020-10-05T09:00:00-04:00",
        "2026-02-02T09:00:00-05:00",
        poll_interval=60.0,
    )

    assert fake.calls[0] == {
        "method": "POST",
        "path": "/etl/news/backfill",
        "json": {
            "symbols": ["AAPL"],
            "start": "2020-10-05T09:00:00-04:00",
            "end": "2026-02-02T09:00:00-05:00",
        },
    }
    assert heartbeats == ["news-1"]
    assert result.status == "complete"
    assert result.events_scored == 12


@pytest.mark.parametrize(
    ("status_code", "job_status", "message"),
    [
        (404, "running", "was lost"),
        (200, "failed", "failed: provider denied"),
    ],
)
def test_news_backfill_activity_surfaces_retryable_lost_or_failed_jobs(
    monkeypatch, status_code, job_status, message
):
    job_id = "news-failed"
    status_path = f"/etl/news/backfill/{job_id}"
    fake = FakeClient(
        {
            "/etl/news/backfill": {"job_id": job_id, "status": "pending"},
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
        training_module.run_news_backfill(
            ["AAPL"],
            "2020-10-05T09:00:00-04:00",
            "2026-02-02T09:00:00-05:00",
            poll_interval=60.0,
        )

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
            news_backfill_start="2020-10-05T09:00:00-04:00",
            news_backfill_end="2026-02-02T09:00:00-05:00",
        )

    @activity.defn(name="run_news_backfill")
    def mock_refresh(symbols, start, end):
        calls["refresh"] += 1
        return NewsBackfillResponse(
            job_id="news-1",
            status="complete",
            windows_done=1,
            windows_total=1,
            events_scored=0,
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
                    detail="Malformed news coverage",
                    retryable=False,
                )
            ],
        )

    @activity.defn(name="run_news_backfill")
    def mock_refresh(symbols, start, end):
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
    assert "Malformed news coverage" in str(exc_info.value.cause)

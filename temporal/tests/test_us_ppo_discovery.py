"""Temporal schedule and skip-path tests for ppo_discovery."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from temporalio import activity

from models.forecast_email import (
    AlpacaPortfolioResponse,
    WeeklyReportEmailResponse,
    WeeklySummaryResponse,
)
from models.news import MondayDecisionWindowResponse
from models.ppo_discovery import PPOInferenceResponse
from schedules import SCHEDULES, _build_spec, second_sunday_of_month_at
from tests.harness import worker_with_activities
from workflows.us_ppo_discovery_allocation import (
    USPPODiscoveryAllocationWorkflow,
    experience_week_bounds,
)

_NY = ZoneInfo("America/New_York")


def _monday_window(run_date: str) -> MondayDecisionWindowResponse:
    del run_date
    cutoff = datetime(2026, 2, 2, 9, 0, tzinfo=_NY)
    return MondayDecisionWindowResponse(
        cutoff=cutoff,
        start_exclusive=datetime(2026, 1, 26, 9, 0, tzinfo=_NY),
        end_inclusive=cutoff,
    )


@activity.defn(name="get_monday_decision_window")
def mock_get_monday_decision_window(run_date: str) -> MondayDecisionWindowResponse:
    return _monday_window(run_date)


def _schedule(schedule_id: str) -> dict:
    return next(schedule for schedule in SCHEDULES if schedule["id"] == schedule_id)


def test_existing_schedule_ids_unchanged() -> None:
    ids = {item["id"] for item in SCHEDULES}
    assert "us-weekly-allocate" in ids
    assert "us-alpha-hrp" in ids
    assert "us-ppo-discovery-allocate" in ids
    assert "us-ppo-discovery-training" in ids


def test_ppo_weekly_is_monday_9_et() -> None:
    spec = _build_spec(_schedule("us-ppo-discovery-allocate"))
    assert spec.time_zone_name == "America/New_York"
    calendar = spec.calendars[0]
    assert calendar.day_of_week[0].start == 1
    assert calendar.hour[0].start == 9
    assert calendar.minute[0].start == 10


def test_second_sunday_helper_is_days_8_to_14() -> None:
    calendar = second_sunday_of_month_at(0, 1)
    assert calendar.day_of_month[0].start == 8
    assert calendar.day_of_month[0].end == 14
    assert calendar.day_of_week[0].start == 0
    spec = _build_spec(_schedule("us-ppo-discovery-training"))
    assert spec.time_zone_name == "UTC"
    assert spec.calendars[0].day_of_month[0].start == 8


def test_experience_week_end_is_the_following_monday() -> None:
    start, end = experience_week_bounds("2026-08-24")
    assert start == "2026-08-24"
    assert end == "2026-08-31"


def test_state_heartbeat_copies_activity_context() -> None:
    import inspect

    from activities.ppo_discovery_inference import build_ppo_discovery_state

    source = inspect.getsource(build_ppo_discovery_state)
    assert "copy_context" in source
    assert "activity_context.run" in source


def test_training_workflow_uses_preflight_snapshot_and_real_eval() -> None:
    import inspect

    from activities.ppo_discovery_training import _poll_ppo_training_job
    from workflows.us_ppo_discovery_training import USPPODiscoveryTrainingWorkflow

    source = inspect.getsource(USPPODiscoveryTrainingWorkflow.run)
    assert "fetch_halal_new_universe" not in source
    assert 'preflight["sorted_symbols"]' in source
    assert 'train.get("evaluation")' in source
    assert "NEWS_ARCHIVE_START_ISO" in source
    assert 'preflight.get("ready")' in source
    assert '"evaluation": {"candidate": True, "promoted": False}' not in source
    poll = inspect.getsource(_poll_ppo_training_job)
    assert "snapshot_sha256" in poll


def test_first_sunday_schedules_untouched() -> None:
    spec = _build_spec(_schedule("us-forecasters-training"))
    assert spec.calendars[0].day_of_month[0].start == 1
    assert spec.calendars[0].day_of_month[0].end == 7


class TestUSPPODiscoverySkip:
    @pytest.mark.asyncio
    async def test_skip_when_open_orders(self):
        @activity.defn(name="resolve_next_attempt")
        def resolve_next_attempt(run_id, as_of_date, accounts=None) -> int:
            assert accounts == ["ppo_discovery"]
            return 1

        @activity.defn(name="get_ppo_discovery_portfolio")
        def get_ppo_discovery_portfolio():
            return AlpacaPortfolioResponse(
                cash=1000.0, positions=[], open_orders_count=2
            )

        @activity.defn(name="generate_ppo_discovery_summary")
        def generate_ppo_discovery_summary(allocation: PPOInferenceResponse):
            return WeeklySummaryResponse(
                summary={"para_1_overall_summary": "skipped"},
                provider="test",
                model_used="test",
                tokens_used=0,
            )

        @activity.defn(name="send_ppo_discovery_email")
        def send_ppo_discovery_email(
            allocation, summary, as_of, skipped=False, skip_reason=""
        ):
            return WeeklyReportEmailResponse(
                is_success=True, subject="skip", body="skip"
            )

        infer_calls: list = []

        @activity.defn(name="infer_ppo_discovery")
        def infer_ppo_discovery(state, state_digest):
            infer_calls.append(1)
            raise AssertionError("inference must not run on skip")

        activities = [
            mock_get_monday_decision_window,
            resolve_next_attempt,
            get_ppo_discovery_portfolio,
            generate_ppo_discovery_summary,
            send_ppo_discovery_email,
            infer_ppo_discovery,
        ]
        async with worker_with_activities(
            [USPPODiscoveryAllocationWorkflow], activities
        ) as env:
            result = await env.client.execute_workflow(
                USPPODiscoveryAllocationWorkflow.run,
                id="test-ppo-skip",
                task_queue="test-queue",
            )
        assert result["skipped"] is True
        assert result["skip_reason"] == "open_orders"
        assert infer_calls == []

    @pytest.mark.asyncio
    async def test_skip_on_news_veto(self):
        @activity.defn(name="resolve_next_attempt")
        def resolve_next_attempt(run_id, as_of_date, accounts=None) -> int:
            return 1

        @activity.defn(name="get_ppo_discovery_portfolio")
        def get_ppo_discovery_portfolio():
            return AlpacaPortfolioResponse(
                cash=1000.0, positions=[], open_orders_count=0
            )

        @activity.defn(name="build_ppo_discovery_state")
        def build_ppo_discovery_state(as_of, run_id, attempt, weights):
            raise RuntimeError("ppo_discovery state failed: news query incomplete")

        infer_calls: list = []

        @activity.defn(name="infer_ppo_discovery")
        def infer_ppo_discovery(state, state_digest):
            infer_calls.append(1)
            raise AssertionError("inference must not run after news veto")

        @activity.defn(name="generate_ppo_discovery_summary")
        def generate_ppo_discovery_summary(allocation: PPOInferenceResponse):
            return WeeklySummaryResponse(
                summary={"para_1_overall_summary": "news veto"},
                provider="test",
                model_used="test",
                tokens_used=0,
            )

        @activity.defn(name="send_ppo_discovery_email")
        def send_ppo_discovery_email(
            allocation, summary, as_of, skipped=False, skip_reason=""
        ):
            return WeeklyReportEmailResponse(
                is_success=True, subject="skip", body=skip_reason
            )

        activities = [
            mock_get_monday_decision_window,
            resolve_next_attempt,
            get_ppo_discovery_portfolio,
            build_ppo_discovery_state,
            infer_ppo_discovery,
            generate_ppo_discovery_summary,
            send_ppo_discovery_email,
        ]
        async with worker_with_activities(
            [USPPODiscoveryAllocationWorkflow], activities
        ) as env:
            result = await env.client.execute_workflow(
                USPPODiscoveryAllocationWorkflow.run,
                id="test-ppo-news-veto",
                task_queue="test-queue",
            )
        assert result["skipped"] is True
        assert "news" in result["skip_reason"].lower()
        assert infer_calls == []

    @pytest.mark.asyncio
    async def test_skip_when_no_current_artifact(self):
        @activity.defn(name="resolve_next_attempt")
        def resolve_next_attempt(run_id, as_of_date, accounts=None) -> int:
            return 1

        @activity.defn(name="get_ppo_discovery_portfolio")
        def get_ppo_discovery_portfolio():
            return AlpacaPortfolioResponse(
                cash=1000.0, positions=[], open_orders_count=0
            )

        @activity.defn(name="build_ppo_discovery_state")
        def build_ppo_discovery_state(as_of, run_id, attempt, weights):
            return {"state_digest": "sha256:abc", "symbols": []}

        @activity.defn(name="infer_ppo_discovery")
        def infer_ppo_discovery(state, state_digest):
            raise RuntimeError("no promoted ppo_discovery artifact")

        @activity.defn(name="generate_ppo_discovery_summary")
        def generate_ppo_discovery_summary(allocation: PPOInferenceResponse):
            return WeeklySummaryResponse(
                summary={"para_1_overall_summary": "no current"},
                provider="test",
                model_used="test",
                tokens_used=0,
            )

        @activity.defn(name="send_ppo_discovery_email")
        def send_ppo_discovery_email(
            allocation, summary, as_of, skipped=False, skip_reason=""
        ):
            return WeeklyReportEmailResponse(
                is_success=True, subject="skip", body=skip_reason
            )

        activities = [
            mock_get_monday_decision_window,
            resolve_next_attempt,
            get_ppo_discovery_portfolio,
            build_ppo_discovery_state,
            infer_ppo_discovery,
            generate_ppo_discovery_summary,
            send_ppo_discovery_email,
        ]
        async with worker_with_activities(
            [USPPODiscoveryAllocationWorkflow], activities
        ) as env:
            result = await env.client.execute_workflow(
                USPPODiscoveryAllocationWorkflow.run,
                id="test-ppo-no-current",
                task_queue="test-queue",
            )
        assert result["skipped"] is True
        assert "promoted" in result["skip_reason"].lower()

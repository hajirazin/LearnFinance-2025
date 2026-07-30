"""Unit tests for production Temporal schedule definitions."""

from types import SimpleNamespace

import pytest
from temporalio.client import ScheduleState

from schedules import (
    SCHEDULES,
    _build_schedule,
    _build_spec,
    _update_existing_schedule,
)


def _schedule(schedule_id: str) -> dict:
    return next(schedule for schedule in SCHEDULES if schedule["id"] == schedule_id)


@pytest.mark.parametrize(
    ("schedule_id", "hour", "minute"),
    [
        ("us-double-hrp", 7, 0),
        ("us-alpha-hrp", 7, 30),
        ("us-weekly-allocate", 8, 0),
        ("us-sac-halal-allocate", 8, 30),
    ],
)
def test_us_weekly_schedules_use_new_york_wall_clock(
    schedule_id: str,
    hour: int,
    minute: int,
):
    spec = _build_spec(_schedule(schedule_id))

    assert spec.time_zone_name == "America/New_York"
    assert len(spec.calendars) == 1
    calendar = spec.calendars[0]
    assert calendar.day_of_week[0].start == 1
    assert calendar.hour[0].start == hour
    assert calendar.minute[0].start == minute


@pytest.mark.parametrize(
    "schedule_id",
    [
        "us-forecasters-training",
        "us-sac-training",
        "us-sac-halal-training",
        "india-monthly-training",
    ],
)
def test_monthly_training_schedule_remains_first_sunday_utc(schedule_id: str):
    spec = _build_spec(_schedule(schedule_id))

    assert spec.time_zone_name == "UTC"
    calendar = spec.calendars[0]
    assert calendar.day_of_month[0].start == 1
    assert calendar.day_of_month[0].end == 7
    assert calendar.day_of_week[0].start == 0


class _FakeScheduleHandle:
    def __init__(self, current_schedule):
        self.current_schedule = current_schedule
        self.update_result = None

    async def update(self, updater):
        update_input = SimpleNamespace(
            description=SimpleNamespace(schedule=self.current_schedule)
        )
        self.update_result = updater(update_input)


class _FakeClient:
    def __init__(self, schedule_id: str, current_schedule):
        self.schedule_id = schedule_id
        self.handle = _FakeScheduleHandle(current_schedule)

    def get_schedule_handle(self, schedule_id: str):
        assert schedule_id == self.schedule_id
        return self.handle


@pytest.mark.asyncio
async def test_existing_schedule_update_replaces_definition_and_preserves_state():
    schedule_config = _schedule("us-weekly-allocate")
    current = _build_schedule(schedule_config)
    paused_state = ScheduleState(paused=True, note="operator pause")
    current.state = paused_state
    client = _FakeClient(schedule_config["id"], current)

    await _update_existing_schedule(client, schedule_config)

    updated = client.handle.update_result.schedule
    assert updated.spec.time_zone_name == "America/New_York"
    assert updated.spec.calendars[0].hour[0].start == 8
    assert updated.state is paused_state

"""Register timezone-aware schedules with Temporal.

Schedule routing is role-based via two task queues:

* ``QUEUE_INFERENCE`` (``learnfinance-inference``) -- weekly allocation
  / HRP workflows. Subscribed by the Pi worker (docker-compose) and
  optionally by a Mac inference worker (``devbox run
  temporal:worker:inference``) as a backup.
* ``QUEUE_TRAINING`` (``learnfinance-training``) -- monthly training
  workflows. Subscribed only by the Mac training worker (``devbox run
  temporal:worker:training``). Training activities are serialized via
  ``TEMPORAL_MAX_CONCURRENT_ACTIVITIES=1`` on that worker so the
  6-hour stagger between training slots cannot double-book the host.

Training schedules use ``ScheduleCalendarSpec`` instead of cron
because Vixie-style cron (which Temporal's ``cron_expressions``
parser follows) OR's day-of-month with day-of-week, making
"first Sunday of the month" impossible to express in a single cron
string. Calendar specs AND all fields, so
``day_of_month=[1..7] AND day_of_week=Sunday`` is exactly the first
Sunday.

Idempotent: safe to run repeatedly (e.g. as a docker compose init
service). Existing schedules are updated in place so timezone, calendar,
workflow, and task-queue changes take effect without a manual delete.
The existing paused/unpaused state is preserved during an update.

Usage:
    cd temporal && uv run python -m schedules
"""

import asyncio
import os
from dataclasses import replace

from temporalio.client import (
    Client,
    Schedule,
    ScheduleActionStartWorkflow,
    ScheduleAlreadyRunningError,
    ScheduleCalendarSpec,
    ScheduleRange,
    ScheduleSpec,
    ScheduleUpdate,
)
from temporalio.contrib.pydantic import pydantic_data_converter

from workflows.india_double_hrp import IndiaDoubleHRPWorkflow
from workflows.india_weekly_allocation import IndiaWeeklyAllocationWorkflow
from workflows.india_weekly_training import IndiaWeeklyTrainingWorkflow
from workflows.us_alpha_hrp import USAlphaHRPWorkflow
from workflows.us_double_hrp import USDoubleHRPWorkflow
from workflows.us_forecasters_training import USForecastersTrainingWorkflow
from workflows.us_sac_halal_allocation import USSACHalalAllocationWorkflow
from workflows.us_sac_halal_training import USSACHalalTrainingWorkflow
from workflows.us_sac_training import USSACTrainingWorkflow
from workflows.us_weekly_allocation import USWeeklyAllocationWorkflow

TEMPORAL_ADDRESS = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")

QUEUE_INFERENCE = "learnfinance-inference"
QUEUE_TRAINING = "learnfinance-training"


def first_sunday_of_month_at(hour: int, minute: int) -> ScheduleCalendarSpec:
    """Calendar spec firing on the first Sunday of each month at HH:MM UTC."""
    return ScheduleCalendarSpec(
        day_of_month=(ScheduleRange(1, 7),),
        day_of_week=(ScheduleRange(0),),
        hour=(ScheduleRange(hour),),
        minute=(ScheduleRange(minute),),
    )


def monday_at(hour: int, minute: int) -> ScheduleCalendarSpec:
    """Calendar spec firing every Monday at HH:MM in the spec timezone."""
    return ScheduleCalendarSpec(
        day_of_week=(ScheduleRange(1),),
        hour=(ScheduleRange(hour),),
        minute=(ScheduleRange(minute),),
    )


SCHEDULES = [
    {
        "id": "us-weekly-allocate",
        "workflow": USWeeklyAllocationWorkflow,
        "workflow_id": "us-weekly-allocate",
        "calendar": monday_at(8, 0),
        "time_zone_name": "America/New_York",
        "task_queue": QUEUE_INFERENCE,
        "description": (
            "US SAC (halal_filtered) weekly allocation Monday 08:00 America/New_York"
        ),
    },
    {
        "id": "india-weekly-allocate",
        "workflow": IndiaWeeklyAllocationWorkflow,
        "workflow_id": "india-weekly-allocate",
        "calendar": monday_at(9, 0),
        "time_zone_name": "Asia/Kolkata",
        "task_queue": QUEUE_INFERENCE,
        "description": "India weekly HRP allocation + email (Monday 9 AM IST)",
    },
    {
        "id": "india-double-hrp",
        "workflow": IndiaDoubleHRPWorkflow,
        "workflow_id": "india-double-hrp",
        "calendar": monday_at(9, 30),
        "time_zone_name": "Asia/Kolkata",
        "task_queue": QUEUE_INFERENCE,
        "description": "India Double HRP (Shariah500 -> top 15) Monday 9:30 AM IST",
    },
    {
        "id": "us-double-hrp",
        "workflow": USDoubleHRPWorkflow,
        "workflow_id": "us-double-hrp",
        "calendar": monday_at(7, 0),
        "time_zone_name": "America/New_York",
        "task_queue": QUEUE_INFERENCE,
        "description": (
            "US Double HRP (halal_new -> sticky top 15) Monday 07:00 America/New_York"
        ),
    },
    {
        "id": "us-alpha-hrp",
        "workflow": USAlphaHRPWorkflow,
        "workflow_id": "us-alpha-hrp",
        "calendar": monday_at(7, 30),
        "time_zone_name": "America/New_York",
        "task_queue": QUEUE_INFERENCE,
        "description": (
            "US Alpha-HRP (PatchTST -> top 15 -> HRP) Monday 07:30 America/New_York"
        ),
    },
    {
        "id": "us-sac-halal-allocate",
        "workflow": USSACHalalAllocationWorkflow,
        "workflow_id": "us-sac-halal-allocate",
        # Parallel A/B sibling of us-weekly-allocate; trades on the
        # dedicated `sac_halal` IBKR account (env
        # IBKR_SAC_HALAL_*, IB Gateway on TCP 4002 paper / 4001 live)
        # via brain_api's /ibkr/* routes -- different broker entirely
        # so client_order_id collisions across the two SAC variants are
        # impossible.
        "calendar": monday_at(8, 30),
        "time_zone_name": "America/New_York",
        "task_queue": QUEUE_INFERENCE,
        "description": (
            "US SAC (halal) weekly allocation (IBKR sac_halal account, "
            "universe=halal) Monday 08:30 America/New_York"
        ),
    },
    # Training schedules -- first Sunday of month, staggered 6h apart
    # starting 00:01 UTC. Routed to QUEUE_TRAINING (Mac-only). The Mac
    # training worker sets TEMPORAL_MAX_CONCURRENT_ACTIVITIES=1 so
    # heavy training activities are serialized even if a run overshoots
    # its 6h slot.
    {
        "id": "us-forecasters-training",
        "workflow": USForecastersTrainingWorkflow,
        "workflow_id": "us-forecasters-training",
        "calendar": first_sunday_of_month_at(0, 1),
        "time_zone_name": "UTC",
        "task_queue": QUEUE_TRAINING,
        "description": (
            "US forecasters training (LSTM + PatchTST) first Sunday of month 00:01 UTC"
        ),
    },
    {
        "id": "us-sac-training",
        "workflow": USSACTrainingWorkflow,
        "workflow_id": "us-sac-training",
        "calendar": first_sunday_of_month_at(6, 1),
        "time_zone_name": "UTC",
        "task_queue": QUEUE_TRAINING,
        "description": (
            "US SAC training (halal_filtered) first Sunday of month 06:01 UTC"
        ),
    },
    {
        "id": "us-sac-halal-training",
        "workflow": USSACHalalTrainingWorkflow,
        "workflow_id": "us-sac-halal-training",
        "calendar": first_sunday_of_month_at(12, 1),
        "time_zone_name": "UTC",
        "task_queue": QUEUE_TRAINING,
        "description": (
            "US SAC training (halal, legacy yfinance universe) "
            "first Sunday of month 12:01 UTC -- parallel A/B "
            "vs sac_halal_filtered"
        ),
    },
    {
        "id": "india-monthly-training",
        "workflow": IndiaWeeklyTrainingWorkflow,
        "workflow_id": "india-monthly-training",
        "calendar": first_sunday_of_month_at(18, 1),
        "time_zone_name": "UTC",
        "task_queue": QUEUE_TRAINING,
        "description": "India PatchTST training first Sunday of month 18:01 UTC",
    },
]


def _build_spec(sched: dict) -> ScheduleSpec:
    return ScheduleSpec(
        calendars=[sched["calendar"]],
        time_zone_name=sched["time_zone_name"],
    )


def _build_schedule(sched: dict) -> Schedule:
    """Build the complete desired Temporal schedule definition."""
    return Schedule(
        action=ScheduleActionStartWorkflow(
            sched["workflow"].run,
            id=sched["workflow_id"],
            task_queue=sched["task_queue"],
        ),
        spec=_build_spec(sched),
    )


async def _update_existing_schedule(client: Client, sched: dict) -> None:
    """Update a schedule definition while preserving its operational state."""
    desired = _build_schedule(sched)
    handle = client.get_schedule_handle(sched["id"])

    def build_update(update_input) -> ScheduleUpdate:
        current_state = update_input.description.schedule.state
        return ScheduleUpdate(schedule=replace(desired, state=current_state))

    await handle.update(build_update)


async def main():
    client = await Client.connect(
        TEMPORAL_ADDRESS, data_converter=pydantic_data_converter
    )

    for sched in SCHEDULES:
        schedule_id = sched["id"]
        desired = _build_schedule(sched)
        try:
            await client.create_schedule(schedule_id, desired)
            print(
                f"  Created: {schedule_id} "
                f"(queue={sched['task_queue']}) - {sched['description']}"
            )
        except ScheduleAlreadyRunningError:
            await _update_existing_schedule(client, sched)
            print(
                f"  Updated: {schedule_id} "
                f"(queue={sched['task_queue']}) - {sched['description']}"
            )

    print(f"\nProcessed {len(SCHEDULES)} schedule(s).")


if __name__ == "__main__":
    asyncio.run(main())

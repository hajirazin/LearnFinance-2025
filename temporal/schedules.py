"""Register cron schedules with Temporal.

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
service). If a schedule already exists, the run logs a loud SKIP and
moves on -- it does NOT update or delete. To change a schedule
(cron / calendar / queue), manually delete it on the server first:

    docker compose exec temporal-server \\
      temporal schedule delete --schedule-id <id> --address 127.0.0.1:7233

then redeploy so this script recreates it with the new config.

Usage:
    cd temporal && uv run python -m schedules
"""

import asyncio
import os

from temporalio.client import (
    Client,
    Schedule,
    ScheduleActionStartWorkflow,
    ScheduleAlreadyRunningError,
    ScheduleCalendarSpec,
    ScheduleRange,
    ScheduleSpec,
)
from temporalio.contrib.pydantic import pydantic_data_converter

from workflows.india_double_hrp import IndiaDoubleHRPWorkflow
from workflows.india_weekly_allocation import IndiaWeeklyAllocationWorkflow
from workflows.india_weekly_training import IndiaWeeklyTrainingWorkflow
from workflows.us_alpha_hrp import USAlphaHRPWorkflow
from workflows.us_double_hrp import USDoubleHRPWorkflow
from workflows.us_forecasters_training import USForecastersTrainingWorkflow
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


SCHEDULES = [
    {
        "id": "us-weekly-allocate",
        "workflow": USWeeklyAllocationWorkflow,
        "workflow_id": "us-weekly-allocate",
        "cron": "0 11 * * 1",  # Monday 11:00 UTC (18:00 IST)
        "task_queue": QUEUE_INFERENCE,
        "description": "US weekly allocation + orders + email (Monday 6 PM IST)",
    },
    {
        "id": "india-weekly-allocate",
        "workflow": IndiaWeeklyAllocationWorkflow,
        "workflow_id": "india-weekly-allocate",
        "cron": "30 3 * * 1",  # Monday 03:30 UTC (09:00 IST)
        "task_queue": QUEUE_INFERENCE,
        "description": "India weekly HRP allocation + email (Monday 9 AM IST)",
    },
    {
        "id": "india-double-hrp",
        "workflow": IndiaDoubleHRPWorkflow,
        "workflow_id": "india-double-hrp",
        "cron": "0 4 * * 1",  # Monday 04:00 UTC (09:30 IST)
        "task_queue": QUEUE_INFERENCE,
        "description": "India Double HRP (Shariah500 -> top 15) Monday 9:30 AM IST",
    },
    {
        "id": "us-double-hrp",
        "workflow": USDoubleHRPWorkflow,
        "workflow_id": "us-double-hrp",
        # 30 minutes after us-weekly-allocate so the two US strategies do
        # not race for brain_api time slots; both still hit Monday close.
        "cron": "30 11 * * 1",  # Monday 11:30 UTC (17:00 IST)
        "task_queue": QUEUE_INFERENCE,
        "description": "US Double HRP (halal_new -> sticky top 15) Monday 5 PM IST",
    },
    {
        "id": "us-alpha-hrp",
        "workflow": USAlphaHRPWorkflow,
        "workflow_id": "us-alpha-hrp",
        # 30 minutes after us-double-hrp so the three US Monday strategies
        # do not contend for brain_api time slots while still hitting the
        # post-close evidence window.
        "cron": "0 12 * * 1",  # Monday 12:00 UTC (17:30 IST)
        "task_queue": QUEUE_INFERENCE,
        "description": "US Alpha-HRP (PatchTST -> top 15 -> HRP) Monday 17:30 IST",
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
        "task_queue": QUEUE_TRAINING,
        "description": "India PatchTST training first Sunday of month 18:01 UTC",
    },
]


def _build_spec(sched: dict) -> ScheduleSpec:
    if "cron" in sched:
        return ScheduleSpec(cron_expressions=[sched["cron"]])
    return ScheduleSpec(calendars=[sched["calendar"]])


async def main():
    client = await Client.connect(
        TEMPORAL_ADDRESS, data_converter=pydantic_data_converter
    )

    for sched in SCHEDULES:
        schedule_id = sched["id"]
        try:
            await client.create_schedule(
                schedule_id,
                Schedule(
                    action=ScheduleActionStartWorkflow(
                        sched["workflow"].run,
                        id=sched["workflow_id"],
                        task_queue=sched["task_queue"],
                    ),
                    spec=_build_spec(sched),
                ),
            )
            print(
                f"  Created: {schedule_id} "
                f"(queue={sched['task_queue']}) - {sched['description']}"
            )
        except ScheduleAlreadyRunningError:
            print(f"  SKIP (already exists, not updating): {schedule_id}")

    print(f"\nProcessed {len(SCHEDULES)} schedule(s).")


if __name__ == "__main__":
    asyncio.run(main())

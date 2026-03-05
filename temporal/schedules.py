"""Register cron schedules with Temporal.

Run once to create all schedules:
    cd temporal && uv run python -m schedules

Re-running is safe -- it deletes and recreates each schedule.
"""

import asyncio
import os

from temporalio.client import (
    Client,
    Schedule,
    ScheduleActionStartWorkflow,
    ScheduleSpec,
)
from temporalio.contrib.pydantic import pydantic_data_converter

from workflows.india_weekly_allocation import IndiaWeeklyAllocationWorkflow
from workflows.india_weekly_training import IndiaWeeklyTrainingWorkflow
from workflows.us_weekly_allocation import USWeeklyAllocationWorkflow
from workflows.us_weekly_training import USWeeklyTrainingWorkflow

TASK_QUEUE = "learnfinance"
TEMPORAL_ADDRESS = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")

SCHEDULES = [
    {
        "id": "us-weekly-allocate",
        "workflow": USWeeklyAllocationWorkflow,
        "workflow_id": "us-weekly-allocate",
        "cron": "0 11 * * 1",  # Monday 11:00 UTC (18:00 IST)
        "description": "US weekly allocation + orders + email (Monday 6 PM IST)",
    },
    {
        "id": "india-weekly-allocate",
        "workflow": IndiaWeeklyAllocationWorkflow,
        "workflow_id": "india-weekly-allocate",
        "cron": "30 3 * * 1",  # Monday 03:30 UTC (09:00 IST)
        "description": "India weekly HRP allocation + email (Monday 9 AM IST)",
    },
    {
        "id": "us-weekly-training",
        "workflow": USWeeklyTrainingWorkflow,
        "workflow_id": "us-weekly-training",
        "cron": "0 11 * * 0",  # Sunday 11:00 UTC
        "description": "US weekly training pipeline (Sunday 11 AM UTC)",
    },
    {
        "id": "india-weekly-training",
        "workflow": IndiaWeeklyTrainingWorkflow,
        "workflow_id": "india-weekly-training",
        "cron": "30 4 * * 0",  # Sunday 04:30 UTC (10:00 IST)
        "description": "India PatchTST training (Sunday 10 AM IST)",
    },
]


async def main():
    client = await Client.connect(
        TEMPORAL_ADDRESS, data_converter=pydantic_data_converter
    )

    for sched in SCHEDULES:
        schedule_id = sched["id"]

        try:
            handle = client.get_schedule_handle(schedule_id)
            await handle.delete()
            print(f"  Deleted existing schedule: {schedule_id}")
        except Exception:
            pass

        await client.create_schedule(
            schedule_id,
            Schedule(
                action=ScheduleActionStartWorkflow(
                    sched["workflow"].run,
                    id=sched["workflow_id"],
                    task_queue=TASK_QUEUE,
                ),
                spec=ScheduleSpec(cron_expressions=[sched["cron"]]),
            ),
        )
        desc = sched["description"]
        print(f"  Created: {schedule_id} ({sched['cron']}) - {desc}")

    print(f"\nAll {len(SCHEDULES)} schedules registered.")


if __name__ == "__main__":
    asyncio.run(main())

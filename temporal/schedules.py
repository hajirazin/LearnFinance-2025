"""Register cron schedules with Temporal.

Idempotent: safe to run repeatedly (e.g. as a docker compose init service).
If a schedule already exists, the run logs a loud SKIP and moves on -- it does
NOT update or delete. To change a schedule (e.g. cron expression), manually
delete it on the server first:

    docker compose exec temporal-server \\
      temporal schedule delete --schedule-id <id> --address 127.0.0.1:7233

then redeploy so this script recreates it with the new config. See
temporal/README.md "Changing a schedule on the Pi" for the full procedure.

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
    ScheduleSpec,
)
from temporalio.contrib.pydantic import pydantic_data_converter

from workflows.india_double_hrp import IndiaDoubleHRPWorkflow
from workflows.india_weekly_allocation import IndiaWeeklyAllocationWorkflow
from workflows.us_weekly_allocation import USWeeklyAllocationWorkflow

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
        "id": "india-double-hrp",
        "workflow": IndiaDoubleHRPWorkflow,
        "workflow_id": "india-double-hrp",
        "cron": "0 4 * * 1",  # Monday 04:00 UTC (09:30 IST)
        "description": "India Double HRP (Shariah500 -> top 15) Monday 9:30 AM IST",
    },
]

# Training schedules are intentionally excluded. The Raspberry Pi (the host
# that runs schedules.py today) cannot afford training workloads. Keep this
# block commented for future use: on a beefier host (Mac/GPU), create a
# separate schedules_mac.py that imports from here and registers all 4.
# Do NOT delete.
# SCHEDULES_MAC = [
#     {
#         "id": "us-weekly-training",
#         "workflow": USWeeklyTrainingWorkflow,
#         "workflow_id": "us-weekly-training",
#         "cron": "0 11 * * 0",  # Sunday 11:00 UTC
#         "description": "US weekly training pipeline (Sunday 11 AM UTC)",
#     },
#     {
#         "id": "india-weekly-training",
#         "workflow": IndiaWeeklyTrainingWorkflow,
#         "workflow_id": "india-weekly-training",
#         "cron": "30 4 * * 0",  # Sunday 04:30 UTC (10:00 IST)
#         "description": "India PatchTST training (Sunday 10 AM IST)",
#     },
# ]


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
                        task_queue=TASK_QUEUE,
                    ),
                    spec=ScheduleSpec(cron_expressions=[sched["cron"]]),
                ),
            )
            print(
                f"  Created: {schedule_id} ({sched['cron']}) - {sched['description']}"
            )
        except ScheduleAlreadyRunningError:
            print(f"  SKIP (already exists, not updating): {schedule_id}")

    print(f"\nProcessed {len(SCHEDULES)} schedule(s).")


if __name__ == "__main__":
    asyncio.run(main())

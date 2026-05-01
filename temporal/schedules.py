"""Register cron schedules with Temporal.

Default ``SCHEDULES`` entries (see list below): ``us-weekly-allocate``,
``india-weekly-allocate``, ``india-double-hrp``, ``us-double-hrp``,
``us-alpha-hrp``. Run ``devbox run temporal:schedule`` once per
environment after deploy.

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
from workflows.us_alpha_hrp import USAlphaHRPWorkflow
from workflows.us_double_hrp import USDoubleHRPWorkflow
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
    {
        "id": "us-double-hrp",
        "workflow": USDoubleHRPWorkflow,
        "workflow_id": "us-double-hrp",
        # 30 minutes after us-weekly-allocate so the two US strategies do
        # not race for brain_api time slots; both still hit Monday close.
        "cron": "30 11 * * 1",  # Monday 11:30 UTC (17:00 IST)
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
        "description": "US Alpha-HRP (PatchTST -> top 15 -> HRP) Monday 17:30 IST",
    },
]

# Training schedules are intentionally excluded. The Raspberry Pi (the host
# that runs schedules.py today) cannot afford training workloads. Keep this
# block commented for future use: on a beefier host (Mac/GPU), create a
# separate schedules_mac.py that imports from here and registers all of them.
# Do NOT delete.
#
# US training is split across three workflows because the host cannot run
# more than one training job at a time:
# - USForecastersTrainingWorkflow (Saturday 11 UTC) trains LSTM then
#   PatchTST serially on halal_new and emails a forecasters-only report.
# - USSACTrainingWorkflow (Sunday 02 UTC) trains SAC on the
#   halal_filtered top-15 (driven by whatever PatchTST 'current'
#   pointer is live at trigger time) and emails a SAC-only report. The
#   gap from Saturday's forecasters slot is comfortably wider than the
#   10h SAC training timeout.
# - USSACHalalTrainingWorkflow (Sunday 13 UTC, 11 h after the
#   halal_filtered slot to fit inside the 10h training timeout with
#   buffer) trains a parallel SAC on the legacy yfinance halal universe
#   (variable size, ~12-15 stocks) for an A/B comparison and emails its
#   own SAC-only report. Each SAC bucket has an independent 'current'
#   pointer; promoting one MUST NOT touch the other.
# SCHEDULES_MAC = [
#     {
#         "id": "us-forecasters-training",
#         "workflow": USForecastersTrainingWorkflow,
#         "workflow_id": "us-forecasters-training",
#         "cron": "0 11 * * 6",  # Saturday 11:00 UTC
#         "description": "US forecasters training (LSTM + PatchTST) Saturday 11 UTC",
#     },
#     {
#         "id": "us-sac-training",
#         "workflow": USSACTrainingWorkflow,
#         "workflow_id": "us-sac-training",
#         "cron": "0 2 * * 0",  # Sunday 02:00 UTC
#         "description": "US SAC training (halal_filtered) Sunday 02 UTC",
#     },
#     {
#         "id": "us-sac-halal-training",
#         "workflow": USSACHalalTrainingWorkflow,
#         "workflow_id": "us-sac-halal-training",
#         "cron": "0 13 * * 0",  # Sunday 13:00 UTC
#         "description": (
#             "US SAC training (halal) Sunday 13 UTC -- parallel A/B "
#             "vs sac_halal_filtered, runs 11 h after that slot"
#         ),
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

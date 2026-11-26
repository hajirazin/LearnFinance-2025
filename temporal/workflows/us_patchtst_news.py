"""Scheduled precomputation and publication of US PatchTST news features."""

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from zoneinfo import ZoneInfo

    from activities.patchtst_news import refresh_patchtst_news


@workflow.defn
class USPatchTSTNewsWorkflow:
    @workflow.run
    async def run(self) -> dict:
        session_date = workflow.now().astimezone(ZoneInfo("America/New_York")).date()
        return await workflow.execute_activity(
            refresh_patchtst_news,
            session_date.isoformat(),
            start_to_close_timeout=timedelta(hours=4),
            heartbeat_timeout=timedelta(minutes=5),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

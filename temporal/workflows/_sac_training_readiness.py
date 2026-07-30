"""Durable readiness gate shared by monthly SAC training workflows."""

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError

with workflow.unsafe.imports_passed_through():
    from activities.training import preflight_sac_training, refresh_training_data
    from models import RefreshTrainingDataResponse, SACTrainingReadiness

PREFLIGHT_TIMEOUT = timedelta(minutes=5)
REFRESH_TIMEOUT = timedelta(hours=10)
READINESS_RETRY_DAYS = 7


async def await_sac_training_readiness(
    universe: str,
    *,
    force: bool,
) -> tuple[SACTrainingReadiness, RefreshTrainingDataResponse | None, int]:
    """Wait durably for a complete SAC training dataset, for at most seven days."""
    latest_refresh = None
    retry_policy = RetryPolicy(maximum_attempts=2)
    for attempt in range(1, READINESS_RETRY_DAYS + 1):
        readiness = await workflow.execute_activity(
            preflight_sac_training,
            args=[universe, force],
            start_to_close_timeout=PREFLIGHT_TIMEOUT,
            retry_policy=retry_policy,
        )
        if readiness.ready:
            return readiness, latest_refresh, attempt
        non_retryable_issues = [
            issue
            for issue in [*readiness.missing, *readiness.errors]
            if not issue.retryable
        ]
        if non_retryable_issues:
            issues = [issue.model_dump(mode="json") for issue in non_retryable_issues]
            raise ApplicationError(
                f"SAC training readiness has non-retryable issues: {issues}",
                non_retryable=True,
            )
        if attempt == READINESS_RETRY_DAYS:
            issues = [
                issue.model_dump(mode="json")
                for issue in [*readiness.missing, *readiness.errors]
            ]
            raise ApplicationError(
                f"SAC training readiness deadline exceeded: {issues}",
                non_retryable=True,
            )
        latest_refresh = await workflow.execute_activity(
            refresh_training_data,
            args=[universe],
            start_to_close_timeout=REFRESH_TIMEOUT,
            retry_policy=retry_policy,
        )
        await workflow.sleep(timedelta(days=1))
    raise AssertionError("unreachable SAC readiness loop exit")

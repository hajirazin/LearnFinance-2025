"""PPO inference leg only; no order, experience, reporting or email activities."""

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from activities.inference import get_monday_decision_window
    from activities.ppo_discovery_inference import (
        build_ppo_discovery_state,
        get_ppo_discovery_portfolio,
        infer_ppo_discovery,
    )
    from workflows.us_ppo_discovery_allocation import _portfolio_weights


@workflow.defn
class PPOInferenceOnlyDiagnostic:
    @workflow.run
    async def run(self, run_date: str) -> dict:
        retry = RetryPolicy(maximum_attempts=1)
        window = await workflow.execute_activity(
            get_monday_decision_window,
            args=[run_date],
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=retry,
        )
        portfolio = await workflow.execute_activity(
            get_ppo_discovery_portfolio,
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=retry,
        )
        state = await workflow.execute_activity(
            build_ppo_discovery_state,
            args=[
                window.cutoff.isoformat(),
                "diagnostic:ppo:" + workflow.info().workflow_id,
                1,
                _portfolio_weights(portfolio),
            ],
            start_to_close_timeout=timedelta(hours=2),
            heartbeat_timeout=timedelta(seconds=60),
            retry_policy=retry,
        )
        allocation = await workflow.execute_activity(
            infer_ppo_discovery,
            args=[state, state["state_digest"]],
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=retry,
        )
        return allocation.model_dump()

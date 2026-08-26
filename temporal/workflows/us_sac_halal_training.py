"""US SAC weekly training workflow on the legacy ``halal`` universe.

Sibling of :mod:`workflows.us_sac_training`. Both workflows train SAC
on first-Sunday calendar slots but on different universes for an A/B
comparison; only one trainer at a time fits on the host:

* ``USSACTrainingWorkflow``        -- universe ``halal_filtered`` (06:01 UTC)
* ``USSACHalalTrainingWorkflow``   -- universe ``halal``          (12:01 UTC)

The slates are independent: ``halal_filtered`` is the sticky-15 derived
from PatchTST scores on top of ``halal_new``, while ``halal`` is the
legacy yfinance ETF top-holdings (SPUS / HLAL / SPTE) deduplicated and
US-filtered. Symbol count for ``halal`` fluctuates month-to-month
(typical range 12-15), so SAC's actor/critic dim and ``target_entropy``
are resized at training time by the bucket-level config factory in
brain_api -- this workflow just forwards ``universe="halal"`` and lets
the endpoint handle sizing.

Steps:

1. Fetch ``halal`` universe (yfinance ETF top-holdings, monthly cache).
2. Run durable readiness preflight; refresh and retry daily for up to 7 days.
3. Train SAC on ``halal`` once ready.
4. Generate SAC-only LLM summary (forwards ``universe="halal"``).
5. Send SAC-only email (subject becomes "US SAC (halal) Training: ...").
"""

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

from workflows._sac_training_readiness import await_sac_training_readiness

with workflow.unsafe.imports_passed_through():
    from activities.training import (
        fetch_halal_universe,
        generate_sac_training_summary,
        send_sac_training_email,
        train_sac,
    )
    from models import SACTrainingWorkflowInput

SHORT_TIMEOUT = timedelta(minutes=5)
TRAINING_TIMEOUT = timedelta(hours=10)
HEARTBEAT_TIMEOUT = timedelta(minutes=10)


@workflow.defn
class USSACHalalTrainingWorkflow:
    @workflow.run
    async def run(self, request: SACTrainingWorkflowInput | None = None) -> dict:
        request = request or SACTrainingWorkflowInput()
        workflow.logger.info("Starting US SAC (halal) training pipeline...")

        halal_result = await workflow.execute_activity(
            fetch_halal_universe,
            start_to_close_timeout=SHORT_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=2),
        )

        (
            readiness,
            refresh_result,
            preflight_attempts,
        ) = await await_sac_training_readiness(
            "halal",
            force=request.force,
        )

        sac_result = await workflow.execute_activity(
            train_sac,
            args=["halal", request.force],
            start_to_close_timeout=TRAINING_TIMEOUT,
            heartbeat_timeout=HEARTBEAT_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=2),
        )

        summary_result = await workflow.execute_activity(
            generate_sac_training_summary,
            args=[sac_result, "halal"],
            start_to_close_timeout=SHORT_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=2),
        )

        email_result = await workflow.execute_activity(
            send_sac_training_email,
            args=[sac_result, summary_result, "halal"],
            start_to_close_timeout=SHORT_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=2),
        )

        workflow.logger.info("US SAC (halal) training pipeline complete!")

        return {
            "halal": {
                "stocks": len(halal_result.get("stocks", [])),
                "total_stocks": halal_result.get("total_stocks"),
            },
            "refresh": None
            if refresh_result is None
            else {
                "job_id": refresh_result.job_id,
                "status": refresh_result.status,
                "windows_done": refresh_result.windows_done,
                "windows_total": refresh_result.windows_total,
                "events_scored": refresh_result.events_scored,
            },
            "readiness": {
                "ready": readiness.ready,
                "attempts": preflight_attempts,
            },
            "sac": {
                "version": sac_result.version,
                "promoted": sac_result.promoted,
                "failure_reasons": sac_result.failure_reasons,
                "force": request.force,
            },
            "summary": {
                "provider": summary_result.provider,
                "model_used": summary_result.model_used,
                "content": summary_result.summary,
            },
            "email": {
                "is_success": email_result.is_success,
                "subject": email_result.subject,
            },
        }

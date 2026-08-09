"""US SAC weekly training workflow.

Targets the first-Sunday 06:01 UTC calendar slot (see
``temporal/schedules.py``). Training activities are serialized on the
Mac worker so they never overlap with forecaster training. Steps:

1. Fetch ``halal_filtered`` universe (PatchTST forecast -> top 15) --
   uses whatever ``current`` PatchTST pointer is live at trigger time,
   so reruns of this workflow do NOT retrain forecasters.
2. Run durable readiness preflight; refresh and retry daily for up to 7 days.
3. Train SAC on ``halal_filtered`` once ready.
4. Generate SAC-only LLM summary.
5. Send SAC-only email.
"""

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

from workflows._sac_training_readiness import await_sac_training_readiness

with workflow.unsafe.imports_passed_through():
    from activities.training import (
        fetch_halal_filtered_universe,
        generate_sac_training_summary,
        send_sac_training_email,
        train_sac,
    )
    from models import SACTrainingWorkflowInput

SHORT_TIMEOUT = timedelta(minutes=5)
TRAINING_TIMEOUT = timedelta(hours=10)
HEARTBEAT_TIMEOUT = timedelta(minutes=10)


@workflow.defn
class USSACTrainingWorkflow:
    @workflow.run
    async def run(self, request: SACTrainingWorkflowInput | None = None) -> dict:
        request = request or SACTrainingWorkflowInput()
        workflow.logger.info("Starting US SAC training pipeline...")

        filtered_result = await workflow.execute_activity(
            fetch_halal_filtered_universe,
            start_to_close_timeout=SHORT_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=2),
        )

        (
            readiness,
            refresh_result,
            preflight_attempts,
        ) = await await_sac_training_readiness(
            "halal_filtered",
            force=request.force,
        )

        sac_result = await workflow.execute_activity(
            train_sac,
            args=["halal_filtered", request.force],
            start_to_close_timeout=TRAINING_TIMEOUT,
            heartbeat_timeout=HEARTBEAT_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=2),
        )

        summary_result = await workflow.execute_activity(
            generate_sac_training_summary,
            args=[sac_result, "halal_filtered"],
            start_to_close_timeout=SHORT_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=2),
        )

        email_result = await workflow.execute_activity(
            send_sac_training_email,
            args=[sac_result, summary_result, "halal_filtered"],
            start_to_close_timeout=SHORT_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=2),
        )

        workflow.logger.info("US SAC training pipeline complete!")

        return {
            "filtered": {
                "stocks": len(filtered_result.get("stocks", [])),
                "model_version": filtered_result.get("model_version"),
                "selection_method": filtered_result.get("selection_method"),
            },
            "refresh": {
                "rows_added": refresh_result.rows_added if refresh_result else 0,
                "remaining_gaps": (
                    refresh_result.remaining_gaps if refresh_result else 0
                ),
                "gaps_pre_api_date": (
                    refresh_result.gaps_pre_api_date if refresh_result else 0
                ),
                "duration_seconds": (
                    refresh_result.duration_seconds if refresh_result else 0.0
                ),
                "hf_url": refresh_result.hf_url if refresh_result else None,
                "published": refresh_result.published if refresh_result else False,
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

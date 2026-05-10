"""US SAC weekly training workflow.

Targets a Sunday cron slot (see ``temporal/schedules.py`` for the
production cadence; runs 12+ hours after the forecasters workflow so it
never overlaps with LSTM/PatchTST training on the same host). Steps:

1. Fetch ``halal_filtered`` universe (PatchTST forecast -> top 15) --
   uses whatever ``current`` PatchTST pointer is live at trigger time,
   so reruns of this workflow do NOT retrain forecasters.
2. Refresh training data (signals for the filtered 15 only).
3. Train SAC on ``halal_filtered``.
4. Generate SAC-only LLM summary.
5. Send SAC-only email.
"""

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from activities.training import (
        fetch_halal_filtered_universe,
        generate_sac_training_summary,
        refresh_training_data,
        send_sac_training_email,
        train_sac,
    )

SHORT_TIMEOUT = timedelta(minutes=5)
TRAINING_TIMEOUT = timedelta(hours=10)
HEARTBEAT_TIMEOUT = timedelta(minutes=10)


@workflow.defn
class USSACTrainingWorkflow:
    @workflow.run
    async def run(self) -> dict:
        workflow.logger.info("Starting US SAC training pipeline...")

        filtered_result = await workflow.execute_activity(
            fetch_halal_filtered_universe,
            start_to_close_timeout=SHORT_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=2),
        )

        refresh_result = await workflow.execute_activity(
            refresh_training_data,
            args=["halal_filtered"],
            start_to_close_timeout=TRAINING_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=2),
        )

        sac_result = await workflow.execute_activity(
            train_sac,
            args=["halal_filtered"],
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
                "sentiment_gaps_filled": refresh_result.sentiment_gaps_filled,
                "fundamentals_refreshed": len(refresh_result.fundamentals_refreshed),
            },
            "sac": {
                "version": sac_result.version,
                "promoted": sac_result.promoted,
                "failure_reasons": sac_result.failure_reasons,
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

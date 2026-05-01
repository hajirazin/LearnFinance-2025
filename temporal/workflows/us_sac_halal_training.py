"""US SAC weekly training workflow on the legacy ``halal`` universe.

Sibling of :mod:`workflows.us_sac_training`. Both workflows train SAC
on a Sunday cron slot but on different universes for an A/B
comparison; only one trainer at a time fits on the host so the two
slots are 11 hours apart (see ``temporal/schedules.py``):

* ``USSACTrainingWorkflow``        -- universe ``halal_filtered`` (Sun 02 UTC)
* ``USSACHalalTrainingWorkflow``   -- universe ``halal``          (Sun 13 UTC)

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
2. Refresh training data (signals for the halal slate only).
3. Train SAC on ``halal``.
4. Generate SAC-only LLM summary (forwards ``universe="halal"``).
5. Send SAC-only email (subject becomes "US SAC (halal) Training: ...").
"""

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from activities.training import (
        fetch_halal_universe,
        generate_sac_training_summary,
        refresh_training_data,
        send_sac_training_email,
        train_sac,
    )

SHORT_TIMEOUT = timedelta(minutes=5)
TRAINING_TIMEOUT = timedelta(hours=10)
HEARTBEAT_TIMEOUT = timedelta(minutes=10)


@workflow.defn
class USSACHalalTrainingWorkflow:
    @workflow.run
    async def run(self) -> dict:
        workflow.logger.info("Starting US SAC (halal) training pipeline...")

        halal_result = await workflow.execute_activity(
            fetch_halal_universe,
            start_to_close_timeout=SHORT_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=2),
        )

        refresh_result = await workflow.execute_activity(
            refresh_training_data,
            args=["halal"],
            start_to_close_timeout=timedelta(hours=1),
            retry_policy=RetryPolicy(maximum_attempts=2),
        )

        sac_result = await workflow.execute_activity(
            train_sac,
            args=["halal"],
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
            "refresh": {
                "sentiment_gaps_filled": refresh_result.sentiment_gaps_filled,
                "fundamentals_refreshed": len(refresh_result.fundamentals_refreshed),
            },
            "sac": {"version": sac_result.version, "promoted": sac_result.promoted},
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

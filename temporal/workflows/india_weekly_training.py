"""India weekly training workflow.

Runs every Sunday at 04:30 UTC (10:00 AM IST):
1. Fetch NiftyShariah500 universe (~210 NSE India stocks)
2. Train India PatchTST (OHLCV forecaster on all ~210 stocks)
3. Fetch halal_india universe (PatchTST forecast -> top 15)
4. Generate India training summary (LLM-powered analysis)
5. Send India training summary email
"""

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from activities.training import (
        fetch_halal_india_universe,
        fetch_nifty_shariah_500_universe,
        generate_india_training_summary,
        send_india_training_email,
        train_india_patchtst,
    )

SHORT_TIMEOUT = timedelta(minutes=5)
TRAINING_TIMEOUT = timedelta(hours=10)
HEARTBEAT_TIMEOUT = timedelta(minutes=10)


@workflow.defn
class IndiaWeeklyTrainingWorkflow:
    @workflow.run
    async def run(self) -> dict:
        workflow.logger.info("Starting India weekly training pipeline...")

        # Step 1: Fetch NiftyShariah500 universe
        nifty_result = await workflow.execute_activity(
            fetch_nifty_shariah_500_universe,
            start_to_close_timeout=SHORT_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        # Step 2: Train India PatchTST
        patchtst_result = await workflow.execute_activity(
            train_india_patchtst,
            start_to_close_timeout=TRAINING_TIMEOUT,
            heartbeat_timeout=HEARTBEAT_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=2),
        )

        # Step 3: Fetch halal_india (PatchTST forecast -> top 15)
        india_filtered = await workflow.execute_activity(
            fetch_halal_india_universe,
            start_to_close_timeout=SHORT_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=2),
        )

        # Step 4: Generate India training summary
        summary_result = await workflow.execute_activity(
            generate_india_training_summary,
            args=[patchtst_result],
            start_to_close_timeout=SHORT_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=2),
        )

        # Step 5: Send India training summary email
        email_result = await workflow.execute_activity(
            send_india_training_email,
            args=[patchtst_result, summary_result],
            start_to_close_timeout=SHORT_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=2),
        )

        workflow.logger.info("India weekly training pipeline complete!")

        return {
            "nifty_shariah_500": {
                "total_stocks": nifty_result.get(
                    "total_stocks", len(nifty_result.get("stocks", []))
                ),
            },
            "patchtst": {
                "version": patchtst_result.version,
                "promoted": patchtst_result.promoted,
            },
            "halal_india": {
                "stocks": len(india_filtered.get("stocks", [])),
                "model_version": india_filtered.get("model_version"),
                "selection_method": india_filtered.get("selection_method"),
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

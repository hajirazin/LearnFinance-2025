"""US Forecasters (LSTM + PatchTST) weekly training workflow.

Targets a Saturday cron slot (see ``temporal/schedules.py`` for the
production cadence). Steps:

1. Fetch ``halal_new`` universe (~410 symbols, fail fast if scraping
   broken).
2. Train LSTM on ``halal_new`` (pure-price forecaster).
3. Train PatchTST on ``halal_new`` (5-channel OHLCV forecaster).
4. Generate forecasters-only LLM summary (LSTM + PatchTST).
5. Send forecasters-only email.

The two training activities run **strictly serially** (sequential
``await workflow.execute_activity`` calls -- no ``asyncio.gather``)
because the Mac/GPU host cannot fit two concurrent training jobs in
memory. SAC training lives in :class:`USSACTrainingWorkflow`, runs the
following day, and consumes whatever ``current`` PatchTST pointer this
workflow leaves behind.
"""

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from activities.training import (
        fetch_halal_new_universe,
        generate_forecasters_training_summary,
        send_forecasters_training_email,
        train_lstm,
        train_patchtst,
    )

SHORT_TIMEOUT = timedelta(minutes=5)
TRAINING_TIMEOUT = timedelta(hours=10)
HEARTBEAT_TIMEOUT = timedelta(minutes=10)


@workflow.defn
class USForecastersTrainingWorkflow:
    @workflow.run
    async def run(self) -> dict:
        workflow.logger.info("Starting US forecasters training pipeline...")

        halal_new_result = await workflow.execute_activity(
            fetch_halal_new_universe,
            start_to_close_timeout=SHORT_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        # Strictly serial: the host machine cannot run two training
        # jobs concurrently, so we await LSTM before kicking off
        # PatchTST. Do NOT replace this with asyncio.gather.
        lstm_result = await workflow.execute_activity(
            train_lstm,
            args=["halal_new"],
            start_to_close_timeout=TRAINING_TIMEOUT,
            heartbeat_timeout=HEARTBEAT_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=2),
        )
        patchtst_result = await workflow.execute_activity(
            train_patchtst,
            args=["halal_new"],
            start_to_close_timeout=TRAINING_TIMEOUT,
            heartbeat_timeout=HEARTBEAT_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=2),
        )

        summary_result = await workflow.execute_activity(
            generate_forecasters_training_summary,
            args=[lstm_result, patchtst_result],
            start_to_close_timeout=SHORT_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=2),
        )

        email_result = await workflow.execute_activity(
            send_forecasters_training_email,
            args=[lstm_result, patchtst_result, summary_result],
            start_to_close_timeout=SHORT_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=2),
        )

        workflow.logger.info("US forecasters training pipeline complete!")

        return {
            "halal_new": {
                "total_stocks": halal_new_result.get(
                    "total_stocks", len(halal_new_result.get("stocks", []))
                ),
            },
            "lstm": {
                "version": lstm_result.version,
                "promoted": lstm_result.promoted,
            },
            "patchtst": {
                "version": patchtst_result.version,
                "promoted": patchtst_result.promoted,
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

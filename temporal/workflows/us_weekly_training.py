"""US weekly training workflow.

Runs every Sunday at 11 AM UTC:
1. Fetch halal_new universe (~410 symbols, fail fast if scraping broken)
2. Train LSTM (pure price forecaster on all ~410 halal_new)
3. Train PatchTST (OHLCV forecaster on all ~410 halal_new)
4. Fetch halal_filtered universe (PatchTST forecast -> top 15)
5. Refresh training data (signals for filtered 15 only)
6. Train PPO (RL allocator on filtered 15)
7. Train SAC (RL allocator on filtered 15)
8. Generate training summary (LLM-powered analysis)
9. Send training summary email
"""

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from activities.training import (
        fetch_halal_filtered_universe,
        fetch_halal_new_universe,
        generate_training_summary,
        refresh_training_data,
        send_training_summary_email,
        train_lstm,
        train_patchtst,
        train_ppo,
        train_sac,
    )

SHORT_TIMEOUT = timedelta(minutes=5)
TRAINING_TIMEOUT = timedelta(hours=10)
HEARTBEAT_TIMEOUT = timedelta(minutes=10)


@workflow.defn
class USWeeklyTrainingWorkflow:
    @workflow.run
    async def run(self) -> dict:
        workflow.logger.info("Starting weekly training pipeline...")

        # Step 1: Fetch halal_new universe (ensure ~410 symbols cached, fail fast)
        halal_new_result = await workflow.execute_activity(
            fetch_halal_new_universe,
            start_to_close_timeout=SHORT_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        # Steps 2-3: Train forecasters on all ~410 halal_new symbols
        lstm_result = await workflow.execute_activity(
            train_lstm,
            start_to_close_timeout=TRAINING_TIMEOUT,
            heartbeat_timeout=HEARTBEAT_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=2),
        )
        patchtst_result = await workflow.execute_activity(
            train_patchtst,
            start_to_close_timeout=TRAINING_TIMEOUT,
            heartbeat_timeout=HEARTBEAT_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=2),
        )

        # Step 4: Fetch halal_filtered (PatchTST forecast -> top 15)
        filtered_result = await workflow.execute_activity(
            fetch_halal_filtered_universe,
            start_to_close_timeout=SHORT_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=2),
        )

        # Step 5: Refresh training data (signals for filtered 15 only)
        refresh_result = await workflow.execute_activity(
            refresh_training_data,
            start_to_close_timeout=timedelta(hours=1),
            retry_policy=RetryPolicy(maximum_attempts=2),
        )

        # Steps 6-7: Train RL allocators on filtered 15
        ppo_result = await workflow.execute_activity(
            train_ppo,
            start_to_close_timeout=TRAINING_TIMEOUT,
            heartbeat_timeout=HEARTBEAT_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=2),
        )
        sac_result = await workflow.execute_activity(
            train_sac,
            start_to_close_timeout=TRAINING_TIMEOUT,
            heartbeat_timeout=HEARTBEAT_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=2),
        )

        # Step 8: Generate training summary using LLM
        summary_result = await workflow.execute_activity(
            generate_training_summary,
            args=[lstm_result, patchtst_result, ppo_result, sac_result],
            start_to_close_timeout=SHORT_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=2),
        )

        # Step 9: Send training summary email
        email_result = await workflow.execute_activity(
            send_training_summary_email,
            args=[lstm_result, patchtst_result, ppo_result, sac_result, summary_result],
            start_to_close_timeout=SHORT_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=2),
        )

        workflow.logger.info("Weekly training pipeline complete!")

        return {
            "halal_new": {
                "total_stocks": halal_new_result.get(
                    "total_stocks", len(halal_new_result.get("stocks", []))
                ),
            },
            "lstm": {"version": lstm_result.version, "promoted": lstm_result.promoted},
            "patchtst": {
                "version": patchtst_result.version,
                "promoted": patchtst_result.promoted,
            },
            "filtered": {
                "stocks": len(filtered_result.get("stocks", [])),
                "model_version": filtered_result.get("model_version"),
                "selection_method": filtered_result.get("selection_method"),
            },
            "refresh": {
                "sentiment_gaps_filled": refresh_result.sentiment_gaps_filled,
                "fundamentals_refreshed": len(refresh_result.fundamentals_refreshed),
            },
            "ppo": {"version": ppo_result.version, "promoted": ppo_result.promoted},
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

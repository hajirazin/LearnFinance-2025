"""Monthly ppo_discovery training. Candidate only; never auto-promotes."""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

from workflows._run_identity import in_ist

with workflow.unsafe.imports_passed_through():
    from activities.inference import get_monday_decision_window
    from activities.ppo_discovery_reporting import (
        generate_ppo_discovery_training_summary,
        send_ppo_discovery_training_email,
    )
    from activities.ppo_discovery_training import (
        etl_ppo_discovery_news,
        preflight_ppo_discovery_training,
        train_ppo_discovery,
    )
    from models.news import NEWS_ARCHIVE_START_ISO

ACTIVITY_TIMEOUT = timedelta(minutes=30)
TRAIN_TIMEOUT = timedelta(hours=24)
ACTIVITY_RETRY = 2


@workflow.defn
class USPPODiscoveryTrainingWorkflow:
    @workflow.run
    async def run(self) -> dict:
        as_of = in_ist(workflow.now()).date().isoformat()
        experiment_id = f"ppo-discovery-{as_of}"
        preflight = await workflow.execute_activity(
            preflight_ppo_discovery_training,
            args=[as_of, experiment_id],
            start_to_close_timeout=ACTIVITY_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
        )
        symbols = list(preflight["sorted_symbols"])
        snapshot_sha256 = preflight["snapshot_sha256"]
        if not preflight.get("ready"):
            return {
                "as_of": as_of,
                "experiment_id": experiment_id,
                "preflight": preflight,
                "skipped": True,
                "promoted": False,
            }
        decision_window = await workflow.execute_activity(
            get_monday_decision_window,
            args=[as_of],
            start_to_close_timeout=ACTIVITY_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
        )
        await workflow.execute_activity(
            etl_ppo_discovery_news,
            args=[
                NEWS_ARCHIVE_START_ISO,
                decision_window.end_inclusive.isoformat(),
                symbols,
            ],
            start_to_close_timeout=TRAIN_TIMEOUT,
            heartbeat_timeout=timedelta(minutes=5),
            retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
        )
        train = await workflow.execute_activity(
            train_ppo_discovery,
            args=[as_of, experiment_id, snapshot_sha256],
            start_to_close_timeout=TRAIN_TIMEOUT,
            heartbeat_timeout=timedelta(minutes=5),
            retry_policy=RetryPolicy(maximum_attempts=1),
        )
        summary_payload = {
            "version": train.get("version") or train.get("job_id", "pending"),
            "promoted": False,
            "snapshot_sha256": snapshot_sha256,
            "evaluation": train.get("evaluation") or {},
            "failure_reasons": list(train.get("failure_reasons") or []),
        }
        summary = await workflow.execute_activity(
            generate_ppo_discovery_training_summary,
            args=[summary_payload],
            start_to_close_timeout=ACTIVITY_TIMEOUT,
        )
        email_payload = {
            "version": summary_payload["version"],
            "promoted": False,
            "snapshot_sha256": summary_payload["snapshot_sha256"],
            "evaluation": summary_payload["evaluation"],
            "failure_reasons": list(summary_payload.get("failure_reasons") or []),
            "para_1_overall": summary.summary.get("para_1_overall", ""),
            "para_2_metrics": summary.summary.get("para_2_metrics", ""),
            "para_3_recommendations": summary.summary.get("para_3_recommendations", ""),
        }
        email = await workflow.execute_activity(
            send_ppo_discovery_training_email,
            args=[email_payload],
            start_to_close_timeout=ACTIVITY_TIMEOUT,
        )
        return {
            "as_of": as_of,
            "experiment_id": experiment_id,
            "preflight": preflight,
            "train": train,
            "promoted": False,
            "email": {"is_success": email.is_success, "subject": email.subject},
        }

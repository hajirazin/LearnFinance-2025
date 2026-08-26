"""Training activities for ppo_discovery."""

from __future__ import annotations

import logging
import time

from temporalio import activity
from temporalio.exceptions import ApplicationError

from activities.client import get_training_client

logger = logging.getLogger(__name__)


@activity.defn
def etl_ppo_discovery_news(start_date: str, end_date: str) -> dict:
    with get_training_client() as client:
        response = client.post(
            "/etl/ppo-discovery/news-history",
            json={
                "start_date": start_date,
                "end_date": end_date,
                "universe": "halal_new",
                "force": False,
            },
        )
        response.raise_for_status()
    return response.json()


@activity.defn
def preflight_ppo_discovery_training(end_date: str, experiment_id: str) -> dict:
    with get_training_client() as client:
        response = client.post(
            "/train/ppo-discovery/preflight",
            json={
                "universe": "halal_new",
                "end_date": end_date,
                "experiment_id": experiment_id,
            },
        )
        response.raise_for_status()
    return response.json()


def _poll_ppo_training_job(
    end_date: str,
    experiment_id: str,
    *,
    poll_interval: float = 60.0,
) -> dict:
    with get_training_client() as client:
        response = client.post(
            "/train/ppo-discovery/full",
            json={
                "universe": "halal_new",
                "end_date": end_date,
                "experiment_id": experiment_id,
            },
        )
        response.raise_for_status()
        payload = response.json()
        if response.status_code == 200 and payload.get("version"):
            return payload
        job_id = payload["job_id"]
        logger.info("ppo_discovery training job started: %s", job_id)
        while True:
            activity.heartbeat(job_id)
            time.sleep(poll_interval)
            status_resp = client.get(f"/train/status/{job_id}")
            status_resp.raise_for_status()
            status = status_resp.json()
            if status["status"] == "completed":
                result = status.get("result") or {}
                if not result.get("version"):
                    raise ApplicationError(
                        "ppo_discovery job completed without a version"
                    )
                return result
            if status["status"] in ("failed", "cancelled"):
                raise ApplicationError(
                    f"ppo_discovery training {status['status']}: "
                    f"{status.get('error', 'unknown')}"
                )


@activity.defn
def train_ppo_discovery(end_date: str, experiment_id: str) -> dict:
    return _poll_ppo_training_job(end_date, experiment_id)

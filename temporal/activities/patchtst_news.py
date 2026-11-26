"""Run and monitor the Brain API's closed-session news refresh."""

import time

from temporalio import activity
from temporalio.exceptions import ApplicationError

from activities.client import get_training_client
from activities.heartbeat import heartbeat_until_done


@activity.defn
def refresh_patchtst_news(session_date: str) -> dict:
    with get_training_client() as client:
        response = client.post(
            "/etl/news/patchtst", json={"session_date": session_date}
        )
        response.raise_for_status()
        job_id = response.json()["job_id"]
        with heartbeat_until_done(job_id):
            while True:
                response = client.get(f"/etl/news/sync/{job_id}")
                response.raise_for_status()
                state = response.json()
                if state["status"] == "completed":
                    return state["result"]
                if state["status"] in ("failed", "cancelled"):
                    raise ApplicationError(
                        state["error"] or "PatchTST news refresh failed"
                    )
                time.sleep(20)

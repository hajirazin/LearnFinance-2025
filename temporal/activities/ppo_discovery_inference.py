"""Inference and state-building activities for ppo_discovery."""

from __future__ import annotations

import contextvars
import logging
import threading

import httpx
from temporalio import activity

from activities.client import BRAIN_API_URL, get_client
from models.forecast_email import AlpacaPortfolioResponse
from models.ppo_discovery import PPOInferenceResponse

logger = logging.getLogger(__name__)

PPO_STATE_HTTP_TIMEOUT = httpx.Timeout(
    connect=30.0,
    read=7200.0,
    write=30.0,
    pool=30.0,
)
HEARTBEAT_SECONDS = 30.0


@activity.defn
def get_ppo_discovery_portfolio() -> AlpacaPortfolioResponse:
    with get_client() as client:
        response = client.get("/alpaca/portfolio", params={"account": "ppo_discovery"})
        response.raise_for_status()
    return AlpacaPortfolioResponse(**response.json())


@activity.defn
def build_ppo_discovery_state(
    as_of: str,
    run_id: str,
    attempt: int,
    current_weights: dict[str, float],
) -> dict:
    stop = threading.Event()
    activity_context = contextvars.copy_context()

    def _heartbeat() -> None:
        while not stop.wait(HEARTBEAT_SECONDS):
            activity.heartbeat("ppo-discovery-state")

    worker = threading.Thread(
        target=activity_context.run, args=(_heartbeat,), daemon=True
    )
    worker.start()
    activity.heartbeat("ppo-discovery-state")
    try:
        with httpx.Client(
            base_url=BRAIN_API_URL, timeout=PPO_STATE_HTTP_TIMEOUT
        ) as client:
            response = client.post(
                "/signals/ppo-discovery/state",
                json={
                    "as_of": as_of,
                    "run_id": run_id,
                    "attempt": attempt,
                    "current_weights": current_weights,
                    "universe": "halal_new",
                },
            )
        if response.status_code in {422, 503}:
            raise RuntimeError(f"ppo_discovery state failed: {response.text}")
        response.raise_for_status()
        return response.json()
    finally:
        stop.set()
        worker.join(timeout=1.0)


@activity.defn
def infer_ppo_discovery(state: dict, state_digest: str) -> PPOInferenceResponse:
    with get_client() as client:
        response = client.post(
            "/inference/ppo-discovery",
            json={
                "state": state,
                "state_digest": state_digest,
                "universe": "halal_new",
            },
        )
        if response.status_code == 503:
            raise RuntimeError("no promoted ppo_discovery artifact")
        response.raise_for_status()
    return PPOInferenceResponse(**response.json())

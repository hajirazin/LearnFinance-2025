"""Training activities for ppo_discovery."""

from __future__ import annotations

import logging

from temporalio import activity

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


@activity.defn
def train_ppo_discovery(end_date: str, experiment_id: str) -> dict:
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
    return response.json()

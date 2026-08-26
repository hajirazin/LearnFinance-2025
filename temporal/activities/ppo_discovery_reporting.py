"""Reporting activities for ppo_discovery.

Kept out of reporting.py so that file stays under the 600-line cap.
"""

from __future__ import annotations

import logging

from temporalio import activity

from activities.client import get_client
from models.email import TrainingSummaryEmailResponse
from models.forecast_email import WeeklyReportEmailResponse, WeeklySummaryResponse
from models.llm import TrainingSummaryResponse
from models.ppo_discovery import PPOInferenceResponse

logger = logging.getLogger(__name__)


@activity.defn
def generate_ppo_discovery_summary(
    allocation: PPOInferenceResponse,
) -> WeeklySummaryResponse:
    cash = float(allocation.percentage_weights.get("CASH", 0.0))
    with get_client() as client:
        response = client.post(
            "/llm/ppo-discovery-weekly-summary",
            json={
                "universe": allocation.universe,
                "model_version": allocation.model_version,
                "k": allocation.k,
                "cash_weight": cash,
                "selected_symbols": allocation.selected_symbols,
                "percentage_weights": allocation.percentage_weights,
                "state_digest": allocation.state_digest,
                "explanations": allocation.explanations,
            },
        )
        response.raise_for_status()
    return WeeklySummaryResponse(**response.json())


@activity.defn
def send_ppo_discovery_email(
    allocation: PPOInferenceResponse,
    summary: WeeklySummaryResponse,
    as_of: str,
    skipped: bool = False,
    skip_reason: str = "",
) -> WeeklyReportEmailResponse:
    paras = summary.summary
    with get_client() as client:
        response = client.post(
            "/email/ppo-discovery-weekly-report",
            json={
                "universe": allocation.universe,
                "as_of": as_of,
                "model_version": allocation.model_version,
                "k": allocation.k,
                "cash_weight": float(allocation.percentage_weights.get("CASH", 0.0)),
                "percentage_weights": allocation.percentage_weights,
                "para_1_overall_summary": paras.get("para_1_overall_summary", ""),
                "para_2_selection": paras.get("para_2_selection", ""),
                "para_3_risks": paras.get("para_3_risks", ""),
                "para_4_research": paras.get("para_4_research", ""),
                "skipped": skipped,
                "skip_reason": skip_reason,
            },
        )
        response.raise_for_status()
    return WeeklyReportEmailResponse(**response.json())


@activity.defn
def generate_ppo_discovery_training_summary(payload: dict) -> TrainingSummaryResponse:
    with get_client() as client:
        response = client.post("/llm/ppo-discovery-training-summary", json=payload)
        response.raise_for_status()
    return TrainingSummaryResponse(**response.json())


@activity.defn
def send_ppo_discovery_training_email(payload: dict) -> TrainingSummaryEmailResponse:
    with get_client() as client:
        response = client.post("/email/ppo-discovery-training-summary", json=payload)
        response.raise_for_status()
    return TrainingSummaryEmailResponse(**response.json())

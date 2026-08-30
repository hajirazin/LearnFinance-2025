"""Email reports for ppo_discovery. Sibling family — not merged with HRP/SAC."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from jinja2 import Environment, FileSystemLoader, TemplateNotFound
from pydantic import BaseModel

from brain_api.routes.email.gmail import GmailConfigError, send_html_email
from brain_api.routes.email.models import (
    TrainingSummaryEmailResponse,
    WeeklyReportEmailResponse,
)
from brain_api.routes.email.weekly_report import TEMPLATE_DIR

router = APIRouter()


class PPOWeeklyEmailRequest(BaseModel):
    universe: str
    as_of: str
    model_version: str
    k: int
    cash_weight: float
    percentage_weights: dict[str, float]
    para_1_overall_summary: str = ""
    para_2_selection: str = ""
    para_3_risks: str = ""
    para_4_research: str = ""
    skipped: bool = False
    skip_reason: str = ""


class PPOTrainingEmailRequest(BaseModel):
    version: str
    promoted: bool = False
    snapshot_sha256: str
    evaluation: dict[str, Any] | None = None
    failure_reasons: list[str] = []
    para_1_overall: str = ""
    para_2_metrics: str = ""
    para_3_recommendations: str = ""


def _jinja() -> Environment:
    return Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), autoescape=True)


def _send(template_name: str, context: dict[str, Any], subject: str) -> str:
    try:
        html = _jinja().get_template(template_name).render(**context)
    except TemplateNotFound as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    try:
        send_html_email(subject=subject, html_body=html)
    except GmailConfigError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return html


@router.post("/ppo-discovery-weekly-report", response_model=WeeklyReportEmailResponse)
def ppo_discovery_weekly_report(
    request: PPOWeeklyEmailRequest,
) -> WeeklyReportEmailResponse:
    subject = f"US PPO Discovery ({request.universe}) Weekly Portfolio Analysis {request.as_of}"
    body = _send(
        "ppo_discovery_weekly_report_email.html.j2", request.model_dump(), subject
    )
    return WeeklyReportEmailResponse(is_success=True, subject=subject, body=body)


@router.post(
    "/ppo-discovery-training-summary", response_model=TrainingSummaryEmailResponse
)
def ppo_discovery_training_email(
    request: PPOTrainingEmailRequest,
) -> TrainingSummaryEmailResponse:
    subject = f"US PPO Discovery Training (candidate): {request.version}"
    body = _send(
        "ppo_discovery_training_summary_email.html.j2", request.model_dump(), subject
    )
    return TrainingSummaryEmailResponse(is_success=True, subject=subject, body=body)

"""LLM summaries for ppo_discovery. Sibling family — not merged with HRP/SAC."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from jinja2 import Environment, FileSystemLoader, TemplateNotFound
from pydantic import BaseModel

from brain_api.routes.llm.models import TrainingSummaryResponse, WeeklySummaryResponse
from brain_api.routes.llm.providers import (
    LLMProvider,
    get_llm_provider,
    parse_json_response,
)
from brain_api.routes.llm.weekly_summary import TEMPLATE_DIR

router = APIRouter()


class PPOWeeklySummaryRequest(BaseModel):
    universe: str
    model_version: str
    k: int
    cash_weight: float
    selected_symbols: list[str]
    percentage_weights: dict[str, float]
    state_digest: str
    explanations: dict[str, Any]


class PPOTrainingSummaryRequest(BaseModel):
    version: str
    promoted: bool = False
    snapshot_sha256: str
    evaluation: dict[str, Any]
    failure_reasons: list[str] = []


def _jinja() -> Environment:
    return Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), autoescape=False)


def _call(
    template_name: str, context: dict[str, Any], provider: LLMProvider
) -> dict[str, Any]:
    try:
        prompt = _jinja().get_template(template_name).render(**context)
    except TemplateNotFound as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    try:
        llm_response = provider.generate(prompt)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    try:
        summary = parse_json_response(llm_response.content)
    except ValueError:
        summary = {"para_1_overall_summary": llm_response.content[:500]}
    return {
        "summary": {str(k): str(v) for k, v in summary.items()},
        "provider": provider.name,
        "model_used": llm_response.model,
        "tokens_used": llm_response.tokens_used,
    }


@router.post("/ppo-discovery-weekly-summary", response_model=WeeklySummaryResponse)
def ppo_discovery_weekly_summary(
    request: PPOWeeklySummaryRequest,
    provider: LLMProvider = Depends(get_llm_provider),
) -> WeeklySummaryResponse:
    payload = _call(
        "ppo_discovery_weekly_summary_prompt.j2", request.model_dump(), provider
    )
    return WeeklySummaryResponse(**payload)


@router.post("/ppo-discovery-training-summary", response_model=TrainingSummaryResponse)
def ppo_discovery_training_summary(
    request: PPOTrainingSummaryRequest,
    provider: LLMProvider = Depends(get_llm_provider),
) -> TrainingSummaryResponse:
    payload = _call(
        "ppo_discovery_training_summary_prompt.j2", request.model_dump(), provider
    )
    return TrainingSummaryResponse(**payload)

"""LLM endpoints for AI-powered summaries and analysis."""

from fastapi import APIRouter

from .models import (
    AlphaHRPSummaryRequest,
    ForecastersTrainingSummaryRequest,
    SACTrainingSummaryRequest,
    SACWeeklySummaryRequest,
    TrainingSummaryResponse,
    WeeklySummaryResponse,
)
from .ppo_discovery import router as ppo_discovery_router
from .providers import (
    LLMProvider,
    LLMResponse,
    OllamaProvider,
    OpenAIProvider,
    get_llm_provider,
    parse_json_response,
)
from .training_summary import router as training_summary_router
from .weekly_summary import router as weekly_summary_router

# Create combined router
router = APIRouter()

# Include sub-routers
router.include_router(training_summary_router)
router.include_router(weekly_summary_router)
router.include_router(ppo_discovery_router)

__all__ = [
    "AlphaHRPSummaryRequest",
    "ForecastersTrainingSummaryRequest",
    "LLMProvider",
    "LLMResponse",
    "OllamaProvider",
    "OpenAIProvider",
    "SACTrainingSummaryRequest",
    "SACWeeklySummaryRequest",
    "TrainingSummaryResponse",
    "WeeklySummaryResponse",
    "get_llm_provider",
    "parse_json_response",
    "router",
]

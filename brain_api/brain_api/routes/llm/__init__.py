"""LLM endpoints for AI-powered summaries and analysis."""

from fastapi import APIRouter

from .models import (
    TrainingSummaryRequest,
    TrainingSummaryResponse,
)
from .providers import (
    LLMProvider,
    LLMResponse,
    OllamaProvider,
    OpenAIProvider,
    get_llm_provider,
    parse_json_response,
)
from .training_summary import router as training_summary_router

# Create combined router
router = APIRouter()

# Include sub-routers
router.include_router(training_summary_router)

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "OllamaProvider",
    "OpenAIProvider",
    "TrainingSummaryRequest",
    "TrainingSummaryResponse",
    "get_llm_provider",
    "parse_json_response",
    "router",
]

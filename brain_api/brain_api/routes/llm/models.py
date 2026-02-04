"""Request/response models for LLM endpoints."""

from pydantic import BaseModel

from brain_api.routes.allocation import HRPAllocationResponse
from brain_api.routes.inference.models import (
    LSTMInferenceResponse,
    PatchTSTInferenceResponse,
    PPOInferenceResponse,
    SACInferenceResponse,
)
from brain_api.routes.signals.models import (
    FundamentalsResponse,
    NewsSignalResponse,
)
from brain_api.routes.training.models import (
    LSTMTrainResponse,
    PatchTSTTrainResponse,
    PPOTrainResponse,
    SACTrainResponse,
)

# =============================================================================
# Training Summary Models
# =============================================================================


class TrainingSummaryRequest(BaseModel):
    """Request model for POST /llm/training-summary."""

    lstm: LSTMTrainResponse
    patchtst: PatchTSTTrainResponse
    ppo: PPOTrainResponse
    sac: SACTrainResponse


class TrainingSummaryResponse(BaseModel):
    """Response model for POST /llm/training-summary."""

    summary: dict[str, str]  # Paragraph fields from LLM
    provider: str  # "openai" or "ollama"
    model_used: str  # e.g., "gpt-4o-mini" or "llama3.2"
    tokens_used: int | None  # Total tokens (None for OLLAMA)


# =============================================================================
# Weekly Summary Models
# =============================================================================


class WeeklySummaryRequest(BaseModel):
    """Request model for POST /llm/weekly-summary.

    All fields are the exact response types from their respective endpoints.
    This endpoint generates an AI summary of weekly forecast/allocation results.
    Does NOT include Alpaca order results - that's only for the email endpoint.
    """

    lstm: LSTMInferenceResponse  # from POST /inference/lstm
    patchtst: PatchTSTInferenceResponse  # from POST /inference/patchtst
    news: NewsSignalResponse  # from POST /signals/news
    fundamentals: FundamentalsResponse  # from POST /signals/fundamentals
    hrp: HRPAllocationResponse  # from POST /allocation/hrp
    sac: SACInferenceResponse  # from POST /inference/sac
    ppo: PPOInferenceResponse  # from POST /inference/ppo


class WeeklySummaryResponse(BaseModel):
    """Response model for POST /llm/weekly-summary."""

    summary: dict[str, str]  # 8 paragraph fields from LLM
    provider: str  # "openai" or "ollama"
    model_used: str  # e.g., "gpt-4o-mini" or "llama3.2"
    tokens_used: int | None  # Total tokens (None for OLLAMA)

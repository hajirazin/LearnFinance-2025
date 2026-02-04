"""Request/response models for LLM endpoints."""

from pydantic import BaseModel

from brain_api.routes.training.models import (
    LSTMTrainResponse,
    PatchTSTTrainResponse,
    PPOTrainResponse,
    SACTrainResponse,
)

# =============================================================================
# Request/Response Models
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

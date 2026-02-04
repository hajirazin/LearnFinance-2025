"""Request/response models for email endpoints."""

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


class TrainingSummaryEmailRequest(BaseModel):
    """Request model for POST /email/training-summary.

    Contains all training results and the LLM-generated summary.
    Email recipient configuration comes from environment variables.
    """

    lstm: LSTMTrainResponse
    patchtst: PatchTSTTrainResponse
    ppo: PPOTrainResponse
    sac: SACTrainResponse
    summary: dict[str, str]  # LLM-generated paragraphs


class TrainingSummaryEmailResponse(BaseModel):
    """Response model for POST /email/training-summary."""

    is_success: bool
    subject: str
    body: str  # Full HTML body (for debugging/logging)

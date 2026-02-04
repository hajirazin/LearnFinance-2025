"""Request/response models for email endpoints."""

from pydantic import BaseModel

from brain_api.routes.allocation import HRPAllocationResponse
from brain_api.routes.inference.models import (
    LSTMInferenceResponse,
    PatchTSTInferenceResponse,
    PPOInferenceResponse,
    SACInferenceResponse,
)
from brain_api.routes.training.models import (
    LSTMTrainResponse,
    PatchTSTTrainResponse,
    PPOTrainResponse,
    SACTrainResponse,
)

# =============================================================================
# Training Summary Email Models
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


# =============================================================================
# Weekly Report Email Models
# =============================================================================


class AlgorithmOrderResult(BaseModel):
    """Order execution result for a single algorithm (from Alpaca)."""

    orders_submitted: int
    orders_failed: int
    skipped: bool = False


class OrderResultsData(BaseModel):
    """Order execution results from Alpaca for all algorithms."""

    ppo: AlgorithmOrderResult
    sac: AlgorithmOrderResult
    hrp: AlgorithmOrderResult


class WeeklyReportEmailRequest(BaseModel):
    """Request model for POST /email/weekly-report.

    Contains everything needed to render the email.
    Uses exact API response types for allocation/forecast data.
    Email recipient configuration comes from environment variables (TRAINING_EMAIL_TO).
    """

    # AI Summary (from /llm/weekly-summary)
    summary: dict[str, str]

    # Alpaca Results
    order_results: OrderResultsData
    skipped_algorithms: list[str] = []

    # Date Info
    target_week_start: str
    target_week_end: str
    as_of_date: str

    # RL Allocators - reuse exact API response types
    sac: SACInferenceResponse
    ppo: PPOInferenceResponse

    # HRP - reuse exact API response type
    hrp: HRPAllocationResponse

    # Forecasters - reuse exact API response types
    lstm: LSTMInferenceResponse
    patchtst: PatchTSTInferenceResponse


class WeeklyReportEmailResponse(BaseModel):
    """Response model for POST /email/weekly-report."""

    is_success: bool
    subject: str
    body: str  # Full HTML body (for debugging/logging)

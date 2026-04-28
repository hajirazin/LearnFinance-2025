"""Request/response models for LLM endpoints."""

from pydantic import BaseModel

from brain_api.routes.allocation import HRPAllocationResponse
from brain_api.routes.inference.models import (
    LSTMInferenceResponse,
    PatchTSTInferenceResponse,
    SACInferenceResponse,
)
from brain_api.routes.signals.models import (
    FundamentalsResponse,
    NewsSignalResponse,
)
from brain_api.routes.training.models import (
    LSTMTrainResponse,
    PatchTSTTrainResponse,
    SACTrainResponse,
)

# =============================================================================
# Training Summary Models
# =============================================================================


class TrainingSummaryRequest(BaseModel):
    """Request model for POST /llm/training-summary."""

    lstm: LSTMTrainResponse
    patchtst: PatchTSTTrainResponse
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


class WeeklySummaryResponse(BaseModel):
    """Response model for POST /llm/weekly-summary."""

    summary: dict[str, str]  # 8 paragraph fields from LLM
    provider: str  # "openai" or "ollama"
    model_used: str  # e.g., "gpt-4o-mini" or "llama3.2"
    tokens_used: int | None  # Total tokens (None for OLLAMA)


# =============================================================================
# India Weekly Summary Models
# =============================================================================


class IndiaTrainingSummaryRequest(BaseModel):
    """Request model for POST /llm/india-training-summary.

    India trains PatchTST only (no LSTM, no SAC).
    """

    patchtst: PatchTSTTrainResponse


class IndiaWeeklySummaryRequest(BaseModel):
    """Request model for POST /llm/india-weekly-summary.

    India pipeline is HRP-only (no SAC/news/fundamentals).
    The LLM analyzes HRP concentration and diversification.
    """

    hrp: HRPAllocationResponse  # from POST /allocation/hrp
    universe: str  # e.g. "halal_india" -- passed by Temporal for reporting context


# =============================================================================
# Double HRP Summary Models
# =============================================================================


class DoubleHRPSummaryRequest(BaseModel):
    """Request model for POST /llm/india-double-hrp-summary.

    Two-stage HRP: Stage 1 screens the full universe, Stage 2
    re-allocates the top-N selected stocks.
    """

    stage1: HRPAllocationResponse  # full universe, long lookback
    stage2: HRPAllocationResponse  # top-N stocks, short lookback
    universe: str  # e.g. "nifty_shariah_500"
    top_n: int  # e.g. 15

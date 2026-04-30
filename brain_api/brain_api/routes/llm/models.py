"""Request/response models for LLM endpoints."""

from pydantic import BaseModel

from brain_api.routes.allocation import HRPAllocationResponse
from brain_api.routes.alpha_models import AlphaScoreItem
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

__all__ = [
    "AlphaScoreItem",
    "DoubleHRPSummaryRequest",
    "IndiaTrainingSummaryRequest",
    "IndiaWeeklySummaryRequest",
    "TrainingSummaryRequest",
    "TrainingSummaryResponse",
    "USAlphaHRPSummaryRequest",
    "USDoubleHRPSummaryRequest",
    "WeeklySummaryRequest",
    "WeeklySummaryResponse",
]

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


class USDoubleHRPSummaryRequest(BaseModel):
    """Request model for POST /llm/us-double-hrp-summary.

    US two-stage HRP with sticky selection. Stage 1 screens the full
    halal_new universe (~410 stocks); sticky selection picks 15; Stage 2
    re-allocates those 15. The summary helps the human reviewer
    understand why the chosen 15 were chosen.
    """

    stage1: HRPAllocationResponse  # halal_new universe, 756d lookback
    stage2: HRPAllocationResponse  # selected 15, 252d lookback
    universe: str  # e.g. "halal_new"
    top_n: int  # e.g. 15


# =============================================================================
# US Alpha-HRP Summary Models
# =============================================================================


class USAlphaHRPSummaryRequest(BaseModel):
    """Request model for POST /llm/us-alpha-hrp-summary.

    Stage 1 = PatchTST predicted weekly returns over halal_new
    (~410 stocks); rank-band sticky selection picks ``top_n`` (default
    15) with hold threshold ``hold_threshold`` (default 30); Stage 2
    HRP risk-parity sizes the chosen names. The LLM gets the top-25
    Stage 1 scores, the rank-band sticky stats (kept/fillers/evicted),
    and the final HRP weights.
    """

    stage1_top_scores: list[AlphaScoreItem]  # top 25 by PatchTST score
    model_version: str  # PatchTST model version used for stage 1
    predicted_count: int  # how many of requested_count produced valid scores
    requested_count: int  # full halal_new size sent into PatchTST
    selected_symbols: list[str]  # final top_n chosen
    kept_count: int
    fillers_count: int
    evicted_from_previous: dict[str, str] = {}
    previous_year_week_used: str | None = None
    stage2: HRPAllocationResponse  # HRP weights on the chosen top_n
    # Sticky-history partition key (e.g. "halal_new_alpha"). Keeps
    # rank-band sticky rows isolated from US Double HRP's weight-band
    # rows on the same tradable universe.
    universe: str
    top_n: int  # K_in (entry threshold)
    hold_threshold: int  # K_hold (sticky retention threshold)

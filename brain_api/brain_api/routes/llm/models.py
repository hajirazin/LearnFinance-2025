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
    "AlphaHRPSummaryRequest",
    "AlphaScoreItem",
    "DoubleHRPSummaryRequest",
    "IndiaTrainingSummaryRequest",
    "SACWeeklySummaryRequest",
    "TrainingSummaryRequest",
    "TrainingSummaryResponse",
    "USDoubleHRPSummaryRequest",
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
# SAC Weekly Summary Models
# =============================================================================


class SACWeeklySummaryRequest(BaseModel):
    """Request model for POST /llm/sac-weekly-summary.

    All fields are the exact response types from their respective endpoints.
    This endpoint generates an AI summary of the SAC-only weekly run on the
    SAC Alpaca paper account. HRP weekly reporting lives in the dedicated
    ``/llm/us-alpha-hrp-summary`` endpoint and is not included here.
    Does NOT include Alpaca order results - that's only for the email endpoint.
    """

    lstm: LSTMInferenceResponse  # from POST /inference/lstm
    patchtst: PatchTSTInferenceResponse  # from POST /inference/patchtst
    news: NewsSignalResponse  # from POST /signals/news
    fundamentals: FundamentalsResponse  # from POST /signals/fundamentals
    sac: SACInferenceResponse  # from POST /inference/sac


class WeeklySummaryResponse(BaseModel):
    """Response model for POST /llm/sac-weekly-summary (and other LLM summary endpoints)."""

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
# Alpha-HRP Summary Models (shared across US + India)
# =============================================================================


class AlphaHRPSummaryRequest(BaseModel):
    """Request model for POST /llm/{us,india}-alpha-hrp-summary.

    Both markets share an identical Stage 1 (PatchTST alpha screen) ->
    rank-band sticky -> Stage 2 (HRP) pipeline; only the underlying
    universe + trained weights differ. The LLM payload shape is therefore
    one DTO. The ``universe`` field discriminates -- e.g.
    ``halal_new`` for US, ``halal_india`` for India -- and downstream
    prompt copy can branch on it.

    Sticky-history partition keys (``halal_new_alpha``,
    ``halal_india_alpha``) keep rank-band rows isolated from any
    weight-band variant on the same tradable universe -- see
    :mod:`brain_api.core.strategy_partitions`.
    """

    stage1_top_scores: list[AlphaScoreItem]  # top 25 by PatchTST score
    model_version: str  # PatchTST model version used for stage 1
    predicted_count: int  # how many of requested_count produced valid scores
    requested_count: int  # universe size sent into PatchTST
    selected_symbols: list[str]  # final top_n chosen
    kept_count: int
    fillers_count: int
    evicted_from_previous: dict[str, str] = {}
    previous_year_week_used: str | None = None
    stage2: HRPAllocationResponse  # HRP weights on the chosen top_n
    universe: str
    top_n: int  # K_in (entry threshold)
    hold_threshold: int  # K_hold (sticky retention threshold)

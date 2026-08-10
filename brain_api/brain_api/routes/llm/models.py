"""Request/response models for LLM endpoints."""

from pydantic import BaseModel

from brain_api.routes.allocation import HRPAllocationResponse
from brain_api.routes.alpha_models import AlphaScoreItem
from brain_api.routes.inference.models import (
    PatchTSTInferenceResponse,
    SACInferenceResponse,
)
from brain_api.routes.signals.models import NewsSignalResponse
from brain_api.routes.training.models import (
    LSTMTrainResponse,
    PatchTSTTrainResponse,
    SACTrainResponse,
)

__all__ = [
    "AlphaHRPSummaryRequest",
    "AlphaScoreItem",
    "DoubleHRPSummaryRequest",
    "ForecastersTrainingSummaryRequest",
    "IndiaDoubleHRPSummaryRequest",
    "IndiaTrainingSummaryRequest",
    "SACTrainingSummaryRequest",
    "SACWeeklySummaryRequest",
    "TrainingSummaryResponse",
    "USDoubleHRPSummaryRequest",
    "WeeklySummaryResponse",
]

# =============================================================================
# Training Summary Models
# =============================================================================


class ForecastersTrainingSummaryRequest(BaseModel):
    """Request model for POST /llm/forecasters-training-summary.

    Carries the LSTM + PatchTST training results from the US forecasters
    workflow. SAC is summarised separately by the SAC training workflow
    via :class:`SACTrainingSummaryRequest` so each workflow can email
    its own report independently.
    """

    lstm: LSTMTrainResponse
    patchtst: PatchTSTTrainResponse


class SACTrainingSummaryRequest(BaseModel):
    """Request model for POST /llm/sac-training-summary.

    Carries the SAC training result from a US SAC workflow. The SAC
    workflow consumes whatever PatchTST ``current`` pointer is live at
    trigger time, so forecaster metrics are not part of this payload --
    they live in :class:`ForecastersTrainingSummaryRequest`.

    Two parallel A/B SAC workflows share this endpoint:
    ``USSACTrainingWorkflow`` (universe=``halal_filtered``) and
    ``USSACHalalTrainingWorkflow`` (universe=``halal``). The
    ``universe`` field discriminates so the LLM prompt and resulting
    summary identify which bucket the metrics describe; default
    preserves the legacy single-bucket payload shape.
    """

    sac: SACTrainResponse
    universe: str = "halal_filtered"


class TrainingSummaryResponse(BaseModel):
    """Response model for the split training-summary endpoints.

    Used by both ``/llm/forecasters-training-summary`` and
    ``/llm/sac-training-summary`` (and also the India training summary).
    """

    summary: dict[str, str]  # Paragraph fields from LLM
    provider: str  # "openai" or "ollama"
    model_used: str  # e.g., "gpt-5-mini" or "llama3.2"
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

    Two parallel A/B SAC weekly workflows share this endpoint:
    ``USWeeklyAllocationWorkflow`` (universe=``halal_filtered``) and
    ``USSACHalalAllocationWorkflow`` (universe=``halal``). The
    ``universe`` field is mandatory (no default; AGENTS.md "no silent
    fallbacks") so the prompt always labels the section with the
    correct bucket and the LLM cannot conflate the two runs.
    """

    patchtst: PatchTSTInferenceResponse  # from POST /inference/patchtst
    news: NewsSignalResponse  # from POST /signals/news
    sac: SACInferenceResponse  # from POST /inference/sac
    universe: str  # "halal_filtered" or "halal"; mandatory


class WeeklySummaryResponse(BaseModel):
    """Response model for POST /llm/sac-weekly-summary (and other LLM summary endpoints)."""

    summary: dict[str, str]  # paragraph fields from LLM
    provider: str  # "openai" or "ollama"
    model_used: str  # e.g., "gpt-5-mini" or "llama3.2"
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
    """Shared base for POST /llm/{us,india}-double-hrp-summary.

    Both markets share an identical Stage 1 (HRP weight screen) ->
    weight-band sticky -> Stage 2 (HRP) pipeline; only the underlying
    universe differs. The LLM payload shape is therefore one DTO. The
    ``universe`` field discriminates -- e.g. ``halal_new`` for US,
    ``nifty_shariah_500`` for India -- and downstream prompt copy can
    branch on it.

    Sticky-history partition keys (``halal_new``,
    ``halal_india_double_hrp``) keep weight-band rows isolated from any
    other strategy on the same tradable universe -- see
    :mod:`brain_api.core.strategy_partitions`.

    Sticky outcome fields (``kept_count``, ``fillers_count``,
    ``previous_year_week_used``) come from ``StickyTopNResponse``;
    defaults make cold-start runs valid.
    """

    stage1: HRPAllocationResponse  # full universe, long lookback
    stage2: HRPAllocationResponse  # selected top_n, short lookback
    universe: str  # e.g. "halal_new" / "nifty_shariah_500"
    top_n: int  # e.g. 15
    kept_count: int = 0
    fillers_count: int = 0
    previous_year_week_used: str | None = None
    stickiness_threshold_pp: float = 1.0


class IndiaDoubleHRPSummaryRequest(DoubleHRPSummaryRequest):
    """Request model for POST /llm/india-double-hrp-summary.

    Same fields as the shared :class:`DoubleHRPSummaryRequest` base.
    Subclassed for symmetry with :class:`USDoubleHRPSummaryRequest`
    and to give each endpoint its own OpenAPI schema entry.
    """


class USDoubleHRPSummaryRequest(DoubleHRPSummaryRequest):
    """Request model for POST /llm/us-double-hrp-summary.

    Same fields as the shared :class:`DoubleHRPSummaryRequest` base;
    US two-stage HRP runs on ``halal_new`` and trades through the
    ``dhrp`` Alpaca paper account. The summary helps the human reviewer
    understand why the chosen ``top_n`` were chosen and how stable the
    weight-band sticky kept holdings vs prior week.
    """


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

    stage1_top_scores: list[AlphaScoreItem]  # top 30 by PatchTST score
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

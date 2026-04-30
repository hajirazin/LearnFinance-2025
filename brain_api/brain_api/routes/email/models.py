"""Request/response models for email endpoints."""

from pydantic import BaseModel

from brain_api.routes.allocation import HRPAllocationResponse
from brain_api.routes.alpha_models import AlphaScoreItem
from brain_api.routes.inference.models import (
    LSTMInferenceResponse,
    PatchTSTInferenceResponse,
    SACInferenceResponse,
)
from brain_api.routes.training.models import (
    LSTMTrainResponse,
    PatchTSTTrainResponse,
    SACTrainResponse,
)

__all__ = [
    "AlgorithmOrderResult",
    "AlphaScoreItem",
    "DoubleHRPEmailRequest",
    "IndiaAlphaHRPEmailRequest",
    "IndiaTrainingSummaryEmailRequest",
    "IndiaTrainingSummaryEmailResponse",
    "OrderResultsData",
    "SACWeeklyReportEmailRequest",
    "TrainingSummaryEmailRequest",
    "TrainingSummaryEmailResponse",
    "USAlphaHRPEmailRequest",
    "USDoubleHRPEmailRequest",
    "WeeklyReportEmailResponse",
]

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
    sac: SACTrainResponse
    summary: dict[str, str]  # LLM-generated paragraphs


class TrainingSummaryEmailResponse(BaseModel):
    """Response model for POST /email/training-summary."""

    is_success: bool
    subject: str
    body: str  # Full HTML body (for debugging/logging)


# =============================================================================
# SAC Weekly Report Email Models
# =============================================================================


class AlgorithmOrderResult(BaseModel):
    """Order execution result for a single algorithm (from Alpaca)."""

    orders_submitted: int
    orders_failed: int
    skipped: bool = False


class OrderResultsData(BaseModel):
    """Order execution results from Alpaca for the SAC-only weekly path.

    HRP weekly trading runs in the dedicated US Alpha-HRP path and reports
    its order results on its own email endpoint, so this payload only carries
    the SAC account.
    """

    sac: AlgorithmOrderResult


class SACWeeklyReportEmailRequest(BaseModel):
    """Request model for POST /email/sac-weekly-report.

    Contains everything needed to render the SAC-only weekly email.
    Uses exact API response types for allocation/forecast data.
    Email recipient configuration comes from environment variables (TRAINING_EMAIL_TO).
    """

    # AI Summary (from /llm/sac-weekly-summary)
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

    # Forecasters - reuse exact API response types
    lstm: LSTMInferenceResponse
    patchtst: PatchTSTInferenceResponse


class WeeklyReportEmailResponse(BaseModel):
    """Response model for POST /email/sac-weekly-report (and other weekly email endpoints)."""

    is_success: bool
    subject: str
    body: str  # Full HTML body (for debugging/logging)


# =============================================================================
# India Weekly Report Email Models
# =============================================================================


class IndiaTrainingSummaryEmailRequest(BaseModel):
    """Request model for POST /email/india-training-summary.

    India trains PatchTST only.
    """

    patchtst: PatchTSTTrainResponse
    summary: dict[str, str]  # LLM-generated paragraphs


class IndiaTrainingSummaryEmailResponse(BaseModel):
    """Response model for POST /email/india-training-summary."""

    is_success: bool
    subject: str
    body: str


class IndiaAlphaHRPEmailRequest(BaseModel):
    """Request model for POST /email/india-alpha-hrp-report.

    India weekly allocation is structurally "PatchTST top-15 alpha screen on
    Nifty Shariah 500 (the ``halal_india`` universe) -> HRP", the India
    counterpart of the US Alpha-HRP path. Contains the AI summary and the
    HRP allocation data sized over the alpha-screened picks.
    Email recipient configuration comes from environment variables (TRAINING_EMAIL_TO).
    """

    summary: dict[str, str]  # from POST /llm/india-alpha-hrp-summary (3 paragraphs)
    hrp: HRPAllocationResponse
    universe: str  # e.g. "halal_india" -- passed by Temporal for reporting context
    target_week_start: str
    target_week_end: str
    as_of_date: str


# =============================================================================
# Double HRP Report Email Models
# =============================================================================


class DoubleHRPEmailRequest(BaseModel):
    """Request model for POST /email/india-double-hrp-report.

    Two-stage HRP: Stage 1 screens the full universe, Stage 2
    re-allocates the top-N selected stocks. Email shows both stages.
    """

    summary: dict[str, str]  # from POST /llm/india-double-hrp-summary
    stage1: HRPAllocationResponse  # full universe, long lookback
    stage2: HRPAllocationResponse  # top-N stocks, short lookback
    universe: str  # e.g. "nifty_shariah_500"
    top_n: int  # e.g. 15
    target_week_start: str
    target_week_end: str
    as_of_date: str


class USDoubleHRPEmailRequest(BaseModel):
    """Request model for POST /email/us-double-hrp-report.

    US Double HRP with sticky selection. Differs from the India variant
    in two ways:
    - It includes Alpaca order execution results because US Double HRP
      trades through a paper account.
    - It supports a ``skipped`` short-circuit when last week's orders
      were still open at run time.

    On the skip path, ``stage1``/``stage2`` are still required (they will
    typically be the prior-week's snapshot or empty) but the email
    template hides allocation tables.
    """

    summary: dict[str, str]  # from POST /llm/us-double-hrp-summary
    stage1: HRPAllocationResponse  # halal_new universe, long lookback
    stage2: HRPAllocationResponse  # selected 15, short lookback
    universe: str  # e.g. "halal_new"
    top_n: int  # e.g. 15
    target_week_start: str
    target_week_end: str
    as_of_date: str
    order_results: AlgorithmOrderResult | None = None
    skipped: bool = False
    sticky_kept_count: int = 0
    sticky_fillers_count: int = 0
    previous_year_week_used: str | None = None


# =============================================================================
# US Alpha-HRP Email Models
# =============================================================================


class USAlphaHRPEmailRequest(BaseModel):
    """Request model for POST /email/us-alpha-hrp-report.

    US Alpha-HRP weekly report. Stage 1 is PatchTST predicted weekly
    returns over halal_new (alpha screen); rank-band sticky selection
    picks the top ``top_n`` (default 15) with hold threshold
    ``hold_threshold`` (default 30); Stage 2 HRP risk-parity sizes the
    chosen names. On the skip path the template hides allocation tables
    and shows a banner about the open-orders gate.
    """

    summary: dict[str, str]  # from POST /llm/us-alpha-hrp-summary
    stage1_top_scores: list[AlphaScoreItem]  # top 25 by PatchTST score
    model_version: str
    predicted_count: int
    requested_count: int
    selected_symbols: list[str]
    kept_count: int = 0
    fillers_count: int = 0
    evicted_from_previous: dict[str, str] = {}
    previous_year_week_used: str | None = None
    stage2: HRPAllocationResponse  # HRP weights on the chosen top_n
    # Sticky-history partition key for this strategy. The tradable
    # universe is still halal_new; the partition key
    # ``halal_new_alpha`` keeps the rank-band sticky rows isolated
    # from US Double HRP's weight-band sticky rows on the same
    # universe (see brain_api.core.strategy_partitions).
    universe: str
    top_n: int
    hold_threshold: int
    target_week_start: str
    target_week_end: str
    as_of_date: str
    order_results: AlgorithmOrderResult | None = None
    skipped: bool = False

"""Request/response models for email endpoints."""

from pydantic import BaseModel

from brain_api.routes.allocation import HRPAllocationResponse
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
# Weekly Report Email Models
# =============================================================================


class AlgorithmOrderResult(BaseModel):
    """Order execution result for a single algorithm (from Alpaca)."""

    orders_submitted: int
    orders_failed: int
    skipped: bool = False


class OrderResultsData(BaseModel):
    """Order execution results from Alpaca for all algorithms."""

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


class IndiaWeeklyReportEmailRequest(BaseModel):
    """Request model for POST /email/india-weekly-report.

    India pipeline is HRP-only (no SAC/news/fundamentals/orders).
    Contains the AI summary and HRP allocation data.
    Email recipient configuration comes from environment variables (TRAINING_EMAIL_TO).
    """

    summary: dict[str, str]  # from POST /llm/india-weekly-summary (3 paragraphs)
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

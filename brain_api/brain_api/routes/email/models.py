"""Request/response models for email endpoints."""

from pydantic import BaseModel

from brain_api.routes.allocation import HRPAllocationResponse
from brain_api.routes.alpha_models import AlphaScoreItem
from brain_api.routes.inference.models import (
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
    "AlphaHRPEmailRequest",
    "AlphaScoreItem",
    "DoubleHRPEmailRequest",
    "ForecastersTrainingSummaryEmailRequest",
    "IndiaAlphaHRPEmailRequest",
    "IndiaDoubleHRPEmailRequest",
    "IndiaTrainingSummaryEmailRequest",
    "IndiaTrainingSummaryEmailResponse",
    "OrderDetail",
    "OrderResultsData",
    "PriorAllocation",
    "SACTrainingSummaryEmailRequest",
    "SACWeeklyReportEmailRequest",
    "TrainingSummaryEmailResponse",
    "USAlphaHRPEmailRequest",
    "USDoubleHRPEmailRequest",
    "WeeklyReportEmailResponse",
]

# =============================================================================
# Training Summary Email Models
# =============================================================================


class ForecastersTrainingSummaryEmailRequest(BaseModel):
    """Request model for POST /email/forecasters-training-summary.

    Carries the US LSTM + PatchTST training results plus the
    forecasters-only LLM summary. Sent at the end of the US Forecasters
    Temporal workflow (Saturday). Email recipient configuration comes
    from environment variables (``GMAIL_USER``, ``GMAIL_APP_PASSWORD``,
    ``TRAINING_EMAIL_TO``, ``TRAINING_EMAIL_CC``).
    """

    lstm: LSTMTrainResponse
    patchtst: PatchTSTTrainResponse
    summary: dict[str, str]  # LLM-generated paragraphs


class SACTrainingSummaryEmailRequest(BaseModel):
    """Request model for POST /email/sac-training-summary.

    Carries a US SAC training result plus the SAC-only LLM summary.
    Sent at the end of either US SAC Temporal workflow (Sunday):
    ``USSACTrainingWorkflow`` for ``halal_filtered`` and the parallel
    A/B sibling ``USSACHalalTrainingWorkflow`` for ``halal``. The
    ``universe`` field is rendered into the subject line so a human
    reading inbox can immediately distinguish the two reports without
    opening them. Email recipient configuration comes from the same
    environment variables as the forecasters report.
    """

    sac: SACTrainResponse
    summary: dict[str, str]  # LLM-generated paragraphs
    universe: str = "halal_filtered"


class TrainingSummaryEmailResponse(BaseModel):
    """Response model for the split training-summary email endpoints.

    Used by both ``/email/forecasters-training-summary`` and
    ``/email/sac-training-summary``. Same shape as
    :class:`IndiaTrainingSummaryEmailResponse`, kept separate only for
    OpenAPI documentation clarity (each endpoint advertises its own
    response model in the schema).
    """

    is_success: bool
    subject: str
    body: str  # Full HTML body (for debugging/logging)


# =============================================================================
# SAC Weekly Report Email Models
# =============================================================================


class OrderDetail(BaseModel):
    """A single order row rendered in the per-order email table (US-only).

    Carries everything the email needs to show "what was ordered, at
    what price, with what stop-loss reference". The stop-loss is
    display-only (no Alpaca bracket order is submitted) so
    ``stop_loss_*`` fields are advisory; they exist so the operator
    has a manual exit reference.

    ``stop_loss_reason`` is one of:

    - ``"atr14"``       -- happy path, ATR(14)-based stop computed
    - ``"atr_unavailable"`` -- ATR could not be computed (missing
      OHLC history / fetch failure). Per AGENTS.md rule #1 we surface
      this verbatim rather than falling back to a flat percent.
    - ``"sell_no_stop"`` -- this row is a sell; exits don't carry a stop.

    ``submission_status`` reflects the Alpaca/IBKR submission outcome
    so the operator can see at a glance whether each order actually
    landed on the broker (``"submitted"``), was rejected
    (``"failed"``), or was deduped against a prior attempt
    (``"deduped"``).
    """

    symbol: str
    side: str
    qty: float
    current_price: float
    trade_value: float
    stop_loss_price: float | None = None
    stop_loss_distance_pct: float | None = None
    stop_loss_reason: str
    client_order_id: str
    submission_status: str


class PriorAllocation(BaseModel):
    """ "Going Into This Week" snapshot rendered in the email.

    For US strategies this is sourced live from the broker
    (``/alpaca/portfolio`` or ``/ibkr/portfolio``) so failed orders
    surface as missing positions. For India it is sourced from the
    prior week's ``final_allocation_pct`` rows in
    ``stage1_weight_history`` (paper-only -> DB matches reality).

    The shared partial template renders the source label verbatim so
    the operator knows which one they are looking at.
    """

    weights: dict[str, float] = {}
    source_label: str = ""
    as_of: str | None = None


class AlgorithmOrderResult(BaseModel):
    """Order execution result for a single algorithm (from Alpaca)."""

    orders_submitted: int
    orders_failed: int
    skipped: bool = False
    orders: list[OrderDetail] = []


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

    Two parallel A/B SAC weekly workflows share this endpoint:
    ``USWeeklyAllocationWorkflow`` (universe=``halal_filtered``) and
    ``USSACHalalAllocationWorkflow`` (universe=``halal``). The
    ``universe`` field is mandatory (no default; AGENTS.md "no silent
    fallbacks") and renders into the subject line so both reports are
    distinguishable in the inbox.
    """

    summary: dict[str, str]

    order_results: OrderResultsData
    skipped_algorithms: list[str] = []

    target_week_start: str
    target_week_end: str
    as_of_date: str
    universe: str  # "halal_filtered" or "halal"; mandatory; renders into subject

    sac: SACInferenceResponse

    patchtst: PatchTSTInferenceResponse

    # "Going Into This Week" -- live broker snapshot for US (Alpaca for
    # ``sac``, IBKR for ``sac_halal``). Empty for legacy callers.
    prior_allocation: PriorAllocation | None = None


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


# =============================================================================
# Alpha-HRP Email Models (shared across US + India)
# =============================================================================


class AlphaHRPEmailRequest(BaseModel):
    """Shared base for POST /email/{us,india}-alpha-hrp-report.

    Both markets ship the same Stage 1 (PatchTST alpha screen) + rank-band
    sticky + Stage 2 (HRP) data to the email template. The base owns
    every common field; the US subclass adds Alpaca-specific
    ``order_results`` + ``skipped`` since India does not trade.

    Sticky-history partition keys: ``halal_new_alpha`` (US),
    ``halal_india_alpha`` (India). Tradable universe label lives in
    ``universe`` and is shown to the human reader of the email.
    """

    summary: dict[str, str]  # from POST /llm/{us,india}-alpha-hrp-summary
    stage1_top_scores: list[AlphaScoreItem]  # top 30 by PatchTST score
    model_version: str
    predicted_count: int
    requested_count: int
    selected_symbols: list[str]
    kept_count: int = 0
    fillers_count: int = 0
    evicted_from_previous: dict[str, str] = {}
    previous_year_week_used: str | None = None
    stage2: HRPAllocationResponse  # HRP weights on the chosen top_n
    universe: str
    top_n: int
    hold_threshold: int
    target_week_start: str
    target_week_end: str
    as_of_date: str

    # "Going Into This Week" snapshot. Live broker for US, prior-week
    # ``final_allocation_pct`` from the strategy's sticky partition
    # for India. Empty by default so legacy callers still validate.
    prior_allocation: PriorAllocation | None = None


class IndiaAlphaHRPEmailRequest(AlphaHRPEmailRequest):
    """Request model for POST /email/india-alpha-hrp-report.

    Same fields as the shared :class:`AlphaHRPEmailRequest` base. India
    has no Alpaca paper account, so no ``order_results`` / ``skipped``
    fields are added.

    ``paper_allocation`` carries the whole-share conversion of Stage 2
    weights for display in the email table (India-only, no broker).
    """

    paper_allocation: dict | None = None


# =============================================================================
# Double HRP Report Email Models
# =============================================================================


class DoubleHRPEmailRequest(BaseModel):
    """Shared base for POST /email/{us,india}-double-hrp-report.

    Both markets ship the same Stage 1 (HRP weight screen) + weight-band
    sticky + Stage 2 (HRP) data to the email template. The base owns
    every common field; the US subclass adds Alpaca-specific
    ``order_results`` + ``skipped`` since India does not trade.

    Sticky-history partition keys: ``halal_new`` (US),
    ``halal_india_double_hrp`` (India). Tradable universe label lives in
    ``universe`` and is shown to the human reader of the email; the
    storage partition is strategy-named (not universe-named) and never
    appears here -- see ``brain_api.core.strategy_partitions``.

    Field naming aligns with :class:`AlphaHRPEmailRequest` for symmetry
    across the two HRP families: ``kept_count`` / ``fillers_count``
    match the ``StickyTopNResponse`` shape from
    ``/allocation/sticky-top-n``.
    """

    summary: dict[str, str]
    stage1: HRPAllocationResponse  # full universe, long lookback
    stage2: HRPAllocationResponse  # top-N stocks, short lookback
    universe: str  # e.g. "halal_new" / "nifty_shariah_500"
    top_n: int  # e.g. 15
    target_week_start: str
    target_week_end: str
    as_of_date: str
    kept_count: int = 0
    fillers_count: int = 0
    previous_year_week_used: str | None = None
    # Default 1.0pp matches the policy threshold both markets use today;
    # surfaced so the email template can phrase the sticky rule
    # correctly even if the threshold ever moves.
    stickiness_threshold_pp: float = 1.0

    # "Going Into This Week" snapshot. Live broker for US, prior-week
    # ``final_allocation_pct`` from the strategy's sticky partition
    # for India. Empty by default so legacy callers still validate.
    prior_allocation: PriorAllocation | None = None


class IndiaDoubleHRPEmailRequest(DoubleHRPEmailRequest):
    """Request model for POST /email/india-double-hrp-report.

    Same fields as the shared :class:`DoubleHRPEmailRequest` base. India
    has no Alpaca paper account, so no ``order_results`` / ``skipped``
    fields are added.

    ``paper_allocation`` carries the whole-share conversion of Stage 2
    weights for display in the email table (India-only, no broker).
    """

    paper_allocation: dict | None = None


class USDoubleHRPEmailRequest(DoubleHRPEmailRequest):
    """Request model for POST /email/us-double-hrp-report.

    Extends :class:`DoubleHRPEmailRequest` with Alpaca-specific fields
    (``order_results``, ``skipped``); India does not trade so its
    request omits these.

    On the skip path, ``stage1``/``stage2`` are still required (they will
    typically be the prior-week's snapshot or empty) but the email
    template hides allocation tables.
    """

    order_results: AlgorithmOrderResult | None = None
    skipped: bool = False


# =============================================================================
# US Alpha-HRP Email Models
# =============================================================================


class USAlphaHRPEmailRequest(AlphaHRPEmailRequest):
    """Request model for POST /email/us-alpha-hrp-report.

    US Alpha-HRP weekly report. Stage 1 is PatchTST predicted weekly
    returns over halal_new (alpha screen); rank-band sticky selection
    picks the top ``top_n`` (default 15) with hold threshold
    ``hold_threshold`` (default 20); Stage 2 HRP risk-parity sizes the
    chosen names. On the skip path the template hides allocation tables
    and shows a banner about the open-orders gate.

    Extends :class:`AlphaHRPEmailRequest` with Alpaca-specific fields
    (``order_results``, ``skipped``); India does not trade so its
    request omits these.
    """

    order_results: AlgorithmOrderResult | None = None
    skipped: bool = False

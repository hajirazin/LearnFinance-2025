"""Pydantic models for SAC experience persistence and API endpoints."""

from typing import Any

from pydantic import BaseModel, Field


class ExperienceState(BaseModel):
    """Full state at decision time for RL experience."""

    # Per-stock signals
    signals: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description=(
            "Signals per symbol: {AAPL: {news_sentiment: 0.3, momentum_4w: 0.05, ...}}"
        ),
    )

    # Forecaster predictions
    patchtst_forecasts: dict[str, float] = Field(
        default_factory=dict,
        description="PatchTST predicted weekly returns per symbol: {AAPL: 0.015, MSFT: -0.003}",
    )

    # Current portfolio weights (before action)
    current_weights: dict[str, float] = Field(
        default_factory=dict,
        description="Current portfolio weights including CASH: {AAPL: 0.10, MSFT: 0.08, CASH: 0.82}",
    )


class OrderExecutionReport(BaseModel):
    """Execution status for a single order."""

    symbol: str
    side: str  # "buy" or "sell"
    intended_qty: float
    filled_qty: float
    filled_avg_price: float | None = None
    status: str  # "filled", "partial", "expired", "rejected", "not_found"
    client_order_id: str | None = None


class ExperienceRecord(BaseModel):
    """A single experience record for RL (SAC) training.

    Lifecycle:
    1. Store: Called after inference with full state + intended action
    2. Update execution: Called after orders settle with execution_report
    3. Label: Called next week to fill in reward based on actual portfolio
    """

    run_id: str  # e.g., "paper:2026-01-12:sac" (includes model_type)
    week_start: str  # ISO date
    week_end: str  # ISO date
    model_type: str  # "sac"
    model_version: str

    # Bucket discriminator. Optional with ``None`` for backward
    # compatibility with experience JSON files written before this field
    # existed; the labeller falls back to inferring the universe from
    # the run_id prefix in that case (logged at WARNING). New writes
    # MUST set this so the labeller can route to the right Alpaca
    # account without inference.
    universe: str | None = None

    # Full state at decision time
    state: dict[str, Any] | ExperienceState  # Preserve canonical v2 snapshot fields
    state_digest: str | None = None

    # Intended action (what the policy decided)
    intended_action: dict[str, float] = Field(
        default_factory=dict,
        description="Target weights from policy: {AAPL: 0.12, MSFT: 0.10, CASH: 0.78}",
    )
    intended_turnover: float = 0.0

    # Legacy field for backward compatibility
    action: dict[str, float] = Field(
        default_factory=dict,
        description="Deprecated: use intended_action instead",
    )
    turnover: float = 0.0  # Deprecated: use intended_turnover

    # Actual execution (filled by update-execution endpoint)
    actual_weights: dict[str, float] | None = Field(
        None,
        description="Actual portfolio weights after orders settled",
    )
    execution_report: list[OrderExecutionReport] | list[dict] | None = Field(
        None,
        description="Per-order execution status",
    )
    execution_updated_at: str | None = None
    # Total portfolio equity (cash + positions market value) at the time of
    # the post-trade snapshot, in USD. Required by the IBKR-SG cost model
    # in the labeller to convert weight deltas into share counts. Optional
    # only for backward compatibility with experience records written
    # before this field existed; the labeller falls back to the IBKR cost
    # config's default NAV anchor (USD 10k) and logs a WARNING in that
    # case.
    nav_usd: float | None = Field(
        None,
        description=(
            "Post-trade total portfolio equity in USD; used by the IBKR "
            "cost model in the labeller. Should be set by "
            "/experience/update-execution from the broker portfolio "
            "snapshot."
        ),
    )

    # Reward (filled by labeling job)
    reward: float | None = None
    realized_return: float | None = None
    next_state: ExperienceState | dict[str, Any] | None = None
    labeled_at: str | None = None


class LabelExperienceRequest(BaseModel):
    """Request to label experience records with realized rewards."""

    run_id: str | None = Field(
        None,
        description="Specific run ID to label. If None, labels all unlabeled records.",
    )


class LabelExperienceResponse(BaseModel):
    """Response from experience labeling."""

    records_labeled: int
    records_skipped: int  # Already labeled or week not ended
    errors: list[str]


class StoreExperienceRequest(BaseModel):
    """Request to store an experience record with full state."""

    run_id: str
    week_start: str
    week_end: str
    model_type: str  # "sac"
    model_version: str

    # Bucket discriminator. Optional only because legacy callers (and a
    # handful of test fixtures) predate this field; the SAC Temporal
    # workflows always pass it now so /experience/label/sac can route
    # to the right Alpaca account without inferring from run_id.
    universe: str | None = Field(
        None,
        description=(
            "SAC bucket universe (e.g. 'halal_filtered' or 'halal'). "
            "Required for new SAC writes so the labeller can route to "
            "the correct Alpaca account."
        ),
    )

    # Full state at decision time
    state: dict[str, Any] | ExperienceState = Field(
        ...,
        description="Full state with signals, forecasts, current_weights",
    )
    state_digest: str | None = Field(
        None,
        description="SHA-256 digest returned with the canonical SAC decision state",
    )

    # Intended action from policy
    intended_action: dict[str, float] = Field(
        default_factory=dict,
        description="Target weights from policy",
    )
    intended_turnover: float = 0.0

    # Legacy fields for backward compatibility
    action: dict[str, float] = Field(
        default_factory=dict,
        description="Deprecated: use intended_action",
    )
    turnover: float = 0.0


class StoreExperienceResponse(BaseModel):
    """Response from storing experience."""

    record_id: str
    stored: bool
    model_type: str


class IntendedOrder(BaseModel):
    """An order that was intended to be submitted."""

    symbol: str
    qty: float
    side: str  # "buy" or "sell"
    client_order_id: str


class ExecutedOrder(BaseModel):
    """An order from Alpaca order history (raw response)."""

    client_order_id: str
    status: str  # "filled", "partially_filled", "canceled", "expired", etc.
    filled_qty: str | None = None
    filled_avg_price: str | None = None


class UpdateExecutionRequest(BaseModel):
    """Request to update experience with execution report after orders settle.

    Can provide EITHER:
    1. Pre-computed execution_report (legacy)
    2. Raw intended_orders + executed_orders (new - matching done internally)
    """

    run_id: str
    model_type: str  # "sac"

    # Option 1: Pre-computed execution report (legacy)
    execution_report: list[dict] | None = Field(
        None,
        description="Pre-computed per-order execution status (legacy)",
    )

    # Option 2: Raw data for internal matching (new)
    intended_orders: list[IntendedOrder] | list[dict] | None = Field(
        None,
        description="Orders we intended to submit (from /orders/generate)",
    )
    executed_orders: list[ExecutedOrder] | list[dict] | None = Field(
        None,
        description="Raw order history from Alpaca (from /alpaca/order-history)",
    )

    actual_weights: dict[str, float] | None = Field(
        None,
        description="Actual portfolio weights after orders settled",
    )

    nav_usd: float | None = Field(
        None,
        description=(
            "Post-trade total portfolio equity in USD (cash + positions "
            "market_value). Plumbed in from the broker snapshot so the "
            "labeller's IBKR cost model can size shares correctly."
        ),
    )


class UpdateExecutionResponse(BaseModel):
    """Response from updating execution report."""

    run_id: str
    updated: bool
    orders_filled: int
    orders_partial: int
    orders_expired: int

"""Shared request/response models for inference endpoints."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from brain_api.core.lstm import SymbolPrediction as LSTMSymbolPrediction
from brain_api.core.patchtst import SymbolPrediction as PatchTSTSymbolPrediction

# Re-export for backward compatibility
SymbolPrediction = LSTMSymbolPrediction


# ============================================================================
# LSTM models
# ============================================================================


class LSTMInferenceRequest(BaseModel):
    """Request model for LSTM inference endpoint."""

    as_of_date: str | None = Field(
        None,
        description="Reference date for inference (YYYY-MM-DD). Defaults to today.",
    )
    symbols: Annotated[list[str], Field(min_length=1)] | None = Field(
        None,
        description=(
            "Optional symbol list to scope inference. If omitted, uses model metadata symbols."
        ),
    )


class LSTMInferenceResponse(BaseModel):
    """Response model for LSTM inference endpoint."""

    predictions: list[LSTMSymbolPrediction]
    model_version: str
    as_of_date: str  # YYYY-MM-DD
    target_week_start: str  # YYYY-MM-DD (first trading day of target week)
    target_week_end: str  # YYYY-MM-DD (last trading day of target week)


# ============================================================================
# PatchTST models
# ============================================================================


class PatchTSTInferenceRequest(BaseModel):
    """Request model for PatchTST inference endpoint."""

    as_of_date: str | None = Field(
        None,
        description="Reference date for inference (YYYY-MM-DD). Defaults to today.",
    )
    symbols: Annotated[list[str], Field(min_length=1)] | None = Field(
        None,
        description=(
            "Optional symbol list to scope inference. If omitted, uses model metadata symbols."
        ),
    )


class PatchTSTInferenceResponse(BaseModel):
    """Response model for PatchTST inference endpoint."""

    predictions: list[PatchTSTSymbolPrediction]
    model_version: str
    as_of_date: str  # YYYY-MM-DD
    target_week_start: str  # YYYY-MM-DD (first trading day of target week)
    target_week_end: str  # YYYY-MM-DD (last trading day of target week)
    signals_used: list[str]  # List of signal types available


# ============================================================================
# PatchTST score-batch (alpha screen feed for rank-band selection)
# ============================================================================


class PatchTSTScoreBatchRequest(BaseModel):
    """Request model for POST /inference/patchtst/score-batch.

    Wraps PatchTST batch inference with the score-validation policy that
    feeds rank-band sticky selection. ``market`` selects the storage
    backend (US ``halal_new``-trained vs India ``nifty_shariah_500``-
    trained); the inference math is identical, only the trained weights
    differ. Validation invariants live in
    :func:`brain_api.core.patchtst.score_validation.validate_and_collect_finite_scores`.
    """

    market: Literal["us", "india"] = Field(
        ...,
        description=(
            "Which trained PatchTST artifacts to use. 'us' loads from "
            "PatchTSTModelStorage; 'india' loads from "
            "PatchTSTIndiaModelStorage."
        ),
    )
    symbols: Annotated[list[str], Field(min_length=1)] = Field(
        ...,
        description="Symbols to score. Must be non-empty.",
    )
    as_of_date: str | None = Field(
        None,
        description="Reference date for inference (YYYY-MM-DD). Defaults to today.",
    )
    min_predictions: int = Field(
        15,
        ge=1,
        description=(
            "Minimum count of finite scores required. Below this floor "
            "the rank-band invariant is violated and the request is "
            "rejected with 422."
        ),
    )


class PatchTSTScoreBatchResponse(BaseModel):
    """Response model for POST /inference/patchtst/score-batch.

    Mirrors the Temporal-side ``PatchTSTBatchScores`` shape so the
    activity layer can be a pure HTTP wrapper.
    """

    scores: dict[str, float] = Field(
        ...,
        description=(
            "symbol -> predicted_weekly_return_pct, only finite values. "
            "Suitable as direct input to /allocation/rank-band-top-n."
        ),
    )
    model_version: str
    as_of_date: str
    target_week_start: str | None = None
    target_week_end: str | None = None
    requested_count: int = Field(
        ...,
        description="Original number of symbols requested.",
    )
    predicted_count: int = Field(
        ...,
        description="Count of finite scores returned.",
    )
    excluded_symbols: list[str] = Field(
        default_factory=list,
        description=(
            "Symbols whose prediction was None (insufficient history / "
            "missing data). Non-finite predictions never appear here -- "
            "they raise 422 instead."
        ),
    )


# ============================================================================
# Portfolio models (shared by SAC endpoint)
# ============================================================================


class Position(BaseModel):
    """A single position in the portfolio."""

    symbol: str
    market_value: float = Field(..., ge=0)


class PortfolioSnapshot(BaseModel):
    """Current portfolio state from Alpaca or similar broker."""

    cash: float = Field(..., description="Can be negative for margin accounts")
    positions: list[Position] = Field(default_factory=list)


class WeightChange(BaseModel):
    """Weight change for a single symbol."""

    symbol: str
    current_weight: float
    target_weight: float
    change: float


# ============================================================================
# SAC models (unified with dual forecasts: LSTM + PatchTST)
# ============================================================================


class SACInferenceRequest(BaseModel):
    """Request model for SAC inference endpoint (dual forecasts)."""

    portfolio: PortfolioSnapshot = Field(
        ...,
        description="Current portfolio state (cash + positions)",
    )
    as_of_date: str | None = Field(
        None,
        description="Reference date for inference (YYYY-MM-DD). Defaults to today.",
    )


class SACInferenceResponse(BaseModel):
    """Response model for SAC inference endpoint."""

    target_weights: dict[str, float]
    turnover: float
    target_week_start: str  # YYYY-MM-DD
    target_week_end: str  # YYYY-MM-DD
    model_version: str
    weight_changes: list[WeightChange]

"""Shared request/response models for inference endpoints."""

from typing import Annotated

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

"""Shared request/response models for inference endpoints."""

from datetime import date, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from brain_api.core.lstm import SymbolPrediction as LSTMSymbolPrediction
from brain_api.core.patchtst import SymbolPrediction as PatchTSTSymbolPrediction
from brain_api.routes.news.models import NewsWindowResult

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
# SAC models (PatchTST-only forecasts)
# ============================================================================


class SACMarketHistoryRow(BaseModel):
    """One dated raw SPY/VIX row for causal HMM continuation."""

    date: date
    spy_adjusted_close: float = Field(..., gt=0)
    vix_close: float = Field(..., gt=0)


class SACFeatureBundleRequest(BaseModel):
    """Raw point-in-time evidence from which Brain builds SAC v3 features."""

    model_config = ConfigDict(extra="forbid")

    symbols: Annotated[list[str], Field(min_length=1, max_length=30)]
    adjusted_closes: dict[str, list[float]]
    patchtst_forecasts: dict[str, float]
    market_history: list[SACMarketHistoryRow]
    provenance: dict[str, object] = Field(default_factory=dict)


class SACNewsSymbolAudit(BaseModel):
    symbol: str
    sentiment_score: float
    article_count: int
    coverage_status: Literal["complete", "verified_empty"]


class SACNewsAudit(BaseModel):
    as_of: datetime
    start_exclusive: datetime
    end_inclusive: datetime
    per_symbol: list[SACNewsSymbolAudit]


class SACInferenceRequest(BaseModel):
    """Request model for SAC inference endpoint (PatchTST-only forecasts)."""

    portfolio: PortfolioSnapshot = Field(
        ...,
        description="Current portfolio state (cash + positions)",
    )
    as_of: datetime = Field(..., description="Monday 09:00 America/New_York cutoff")
    as_of_date: str | None = Field(
        None,
        description="Reference date for price sessions (YYYY-MM-DD). Defaults to today.",
    )
    news_window: NewsWindowResult
    feature_bundle: SACFeatureBundleRequest = Field(
        ...,
        description=(
            "Raw adjusted-price history, forecasts, and SPY/VIX history. "
            "News is supplied via news_window; Brain owns adapter math."
        ),
    )


class ForcedLiquidationAudit(BaseModel):
    """Position outside the model slate that order generation must liquidate."""

    symbol: str
    market_value: float
    reason: str = "outside_active_sac_symbol_set"


class SACInferenceResponse(BaseModel):
    """Response model for SAC inference endpoint."""

    target_weights: dict[str, float]
    turnover: float
    target_week_start: str  # YYYY-MM-DD
    target_week_end: str  # YYYY-MM-DD
    model_version: str
    weight_changes: list[WeightChange]
    decision_state: dict[str, object] | None = None
    state_digest: str | None = None
    forced_liquidations: list[ForcedLiquidationAudit] = Field(default_factory=list)
    asset_eligibility: dict[str, bool]
    regime_posterior: Annotated[list[float], Field(min_length=3, max_length=3)]
    sac_schema_version: Literal[3]
    architecture: Literal["masked_attention"]
    news_audit: SACNewsAudit


class SkippedSACInferenceResponse(BaseModel):
    """Reporting-only representation of a SAC run skipped for open orders."""

    skipped: Literal[True]
    algorithm: str
    reason: str
    target_weights: dict[str, float] = Field(default_factory=dict)
    turnover: float = 0.0
    model_version: Literal["skipped"] = "skipped"
    target_week_start: str = ""
    target_week_end: str = ""
    weight_changes: list[WeightChange] = Field(default_factory=list)
    decision_state: None = None

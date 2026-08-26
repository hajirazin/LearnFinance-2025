"""Request and response models for signal endpoints."""

from datetime import date

from pydantic import BaseModel, Field

MAX_PRICE_SYMBOLS = 30
MIN_MOMENTUM_LOOKBACK_BARS = 253


class ClosesRequest(BaseModel):
    """Request model for raw daily closes (SAC momentum signals)."""

    symbols: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_PRICE_SYMBOLS,
        description=f"List of ticker symbols (1-{MAX_PRICE_SYMBOLS})",
    )
    as_of_date: str = Field(
        ...,
        description="Decision date (YYYY-MM-DD); closes are as of/before this date",
    )
    lookback_bars: int = Field(
        MIN_MOMENTUM_LOOKBACK_BARS,
        ge=MIN_MOMENTUM_LOOKBACK_BARS,
        description=(
            "Trailing trading-day bars to return per symbol, oldest first. "
            "Defaults to momentum_12_1's requirement (skip 21 + lookback 252)."
        ),
    )


class ClosesResponse(BaseModel):
    """Point-in-time adjusted closes for SAC feature construction."""

    as_of_date: str
    adjusted_closes: dict[str, list[float]]
    provenance: dict[str, object]


class MarketHistoryRequest(BaseModel):
    """Request aligned SPY/VIX observations for causal HMM continuation."""

    start_date: date = Field(
        ...,
        description="First required post-training-cutoff market session",
    )
    as_of_date: date = Field(
        ...,
        description="Pre-open decision date; only earlier completed XNYS sessions",
    )


class MarketHistoryRow(BaseModel):
    """One aligned raw market observation row."""

    date: date
    spy_adjusted_close: float = Field(..., gt=0)
    vix_close: float = Field(..., gt=0)


class MarketHistoryResponse(BaseModel):
    """Gap-sensitive raw market history consumed by Brain's HMM filter."""

    start_date: date
    as_of_date: date
    rows: list[MarketHistoryRow]
    provenance: dict[str, object]

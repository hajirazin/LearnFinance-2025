"""Inference endpoints for ML models."""

from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from brain_api.core.lstm import (
    SymbolPrediction,
    build_inference_features,
    compute_week_boundaries,
    load_prices_yfinance,
    run_inference,
)
from brain_api.storage.local import LocalModelStorage

router = APIRouter()


# ============================================================================
# Request / Response models
# ============================================================================


class LSTMInferenceRequest(BaseModel):
    """Request model for LSTM inference endpoint."""

    symbols: list[str] = Field(
        ...,
        min_length=1,
        description="List of ticker symbols to predict weekly returns for",
    )
    as_of_date: str | None = Field(
        None,
        description="Reference date for inference (YYYY-MM-DD). Defaults to today.",
    )


class LSTMInferenceResponse(BaseModel):
    """Response model for LSTM inference endpoint."""

    predictions: list[SymbolPrediction]
    model_version: str
    as_of_date: str  # YYYY-MM-DD
    target_week_start: str  # YYYY-MM-DD (first trading day of target week)
    target_week_end: str  # YYYY-MM-DD (last trading day of target week)


# ============================================================================
# Dependency injection for testability
# ============================================================================


def get_storage() -> LocalModelStorage:
    """Get the model storage instance."""
    return LocalModelStorage()


def get_as_of_date(request: LSTMInferenceRequest) -> date:
    """Get the as-of date from request or default to today."""
    if request.as_of_date:
        return date.fromisoformat(request.as_of_date)
    return date.today()


# Type aliases for dependency injection
PriceLoader = type(load_prices_yfinance)
WeekBoundaryComputer = type(compute_week_boundaries)


def get_price_loader() -> PriceLoader:
    """Get the price loading function."""
    return load_prices_yfinance


def get_week_boundary_computer() -> WeekBoundaryComputer:
    """Get the week boundary computation function."""
    return compute_week_boundaries


# ============================================================================
# Endpoint
# ============================================================================


@router.post("/lstm", response_model=LSTMInferenceResponse)
def infer_lstm(
    request: LSTMInferenceRequest,
    storage: LocalModelStorage = Depends(get_storage),
    price_loader: PriceLoader = Depends(get_price_loader),
    week_boundary_computer: WeekBoundaryComputer = Depends(get_week_boundary_computer),
) -> LSTMInferenceResponse:
    """Predict weekly returns for the given symbols.

    This endpoint is designed for Monday runs. It:
    1. Loads the current promoted LSTM model
    2. Computes the target week boundaries (holiday-aware)
    3. Fetches price history from yfinance
    4. Builds feature sequences (60 trading days before target week start)
    5. Runs inference and returns predicted weekly returns

    The prediction is for the weekly return = (week_end_close - week_start_open) / week_start_open,
    expressed as a percentage.

    Args:
        request: LSTMInferenceRequest with symbols list

    Returns:
        LSTMInferenceResponse with per-symbol predictions and metadata

    Raises:
        HTTPException 503: if no trained model is available
    """
    # Get as-of date
    as_of = get_as_of_date(request)

    # Compute holiday-aware week boundaries
    week_boundaries = week_boundary_computer(as_of)

    # Load current model artifacts
    try:
        artifacts = storage.load_current_artifacts()
    except ValueError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e),
        ) from None
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model artifacts incomplete: {e}",
        ) from None

    # Calculate how much history we need to fetch
    # We need sequence_length days of data ending before target_week_start
    # Fetch extra buffer for weekends/holidays (2x should be plenty)
    config = artifacts.config
    buffer_days = config.sequence_length * 2 + 30  # Extra 30 days for safety

    # Calculate start date for data fetch
    from datetime import timedelta
    data_start = week_boundaries.target_week_start - timedelta(days=buffer_days)
    # End date is the day before target_week_start (we don't want target week data)
    data_end = week_boundaries.target_week_start - timedelta(days=1)

    # Fetch price data for all symbols
    prices = price_loader(request.symbols, data_start, data_end)

    # Build features for each symbol
    features_list = []
    for symbol in request.symbols:
        prices_df = prices.get(symbol)
        if prices_df is None or prices_df.empty:
            # Symbol not found or no data
            from brain_api.core.lstm import InferenceFeatures
            features_list.append(
                InferenceFeatures(
                    symbol=symbol,
                    features=None,
                    has_enough_history=False,
                    history_days_used=0,
                    data_end_date=None,
                )
            )
        else:
            features = build_inference_features(
                symbol=symbol,
                prices_df=prices_df,
                config=config,
                cutoff_date=week_boundaries.target_week_start,
            )
            features_list.append(features)

    # Run inference
    predictions = run_inference(
        model=artifacts.model,
        feature_scaler=artifacts.feature_scaler,
        features_list=features_list,
        week_boundaries=week_boundaries,
    )

    # Sort predictions: highest gain → highest loss, with null/insufficient-history at the end
    predictions = _sort_predictions(predictions)

    return LSTMInferenceResponse(
        predictions=predictions,
        model_version=artifacts.version,
        as_of_date=as_of.isoformat(),
        target_week_start=week_boundaries.target_week_start.isoformat(),
        target_week_end=week_boundaries.target_week_end.isoformat(),
    )


def _sort_predictions(predictions: list[SymbolPrediction]) -> list[SymbolPrediction]:
    """Sort predictions by predicted_weekly_return_pct descending.

    Predictions with valid returns are sorted highest to lowest.
    Predictions with null returns (insufficient history) are placed at the end.
    """
    # Separate valid and invalid predictions
    valid = [p for p in predictions if p.predicted_weekly_return_pct is not None]
    invalid = [p for p in predictions if p.predicted_weekly_return_pct is None]

    # Sort valid predictions by return (highest first)
    valid_sorted = sorted(
        valid,
        key=lambda p: p.predicted_weekly_return_pct,  # type: ignore[arg-type]
        reverse=True,
    )

    return valid_sorted + invalid


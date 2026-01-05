"""Inference endpoints for ML models."""

import logging
from datetime import date
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from brain_api.core.config import get_hf_model_repo, get_storage_backend
from brain_api.core.lstm import (
    SymbolPrediction as LSTMSymbolPrediction,
    build_inference_features,
    compute_week_boundaries,
    load_prices_yfinance,
    run_inference,
)
from brain_api.core.patchtst import (
    InferenceFeatures as PatchTSTInferenceFeatures,
    PatchTSTConfig,
    SymbolPrediction as PatchTSTSymbolPrediction,
    WeekBoundaries,
    build_inference_features as patchtst_build_inference_features,
    compute_week_boundaries as patchtst_compute_week_boundaries,
    load_historical_fundamentals,
    load_historical_news_sentiment,
    load_prices_yfinance as patchtst_load_prices,
    run_inference as patchtst_run_inference,
)
from brain_api.storage.local import (
    LSTMArtifacts,
    LocalModelStorage,
    PatchTSTArtifacts,
    PatchTSTModelStorage,
)

router = APIRouter()
logger = logging.getLogger(__name__)

# Re-export for backward compatibility
SymbolPrediction = LSTMSymbolPrediction


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

    predictions: list[LSTMSymbolPrediction]
    model_version: str
    as_of_date: str  # YYYY-MM-DD
    target_week_start: str  # YYYY-MM-DD (first trading day of target week)
    target_week_end: str  # YYYY-MM-DD (last trading day of target week)


class PatchTSTInferenceRequest(BaseModel):
    """Request model for PatchTST inference endpoint."""

    symbols: list[str] = Field(
        ...,
        min_length=1,
        description="List of ticker symbols to predict weekly returns for",
    )
    as_of_date: str | None = Field(
        None,
        description="Reference date for inference (YYYY-MM-DD). Defaults to today.",
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

    # Load current model artifacts (try local first, then HuggingFace)
    artifacts = _load_model_artifacts(storage)

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


def _load_model_artifacts(storage: LocalModelStorage) -> LSTMArtifacts:
    """Load model artifacts with HuggingFace fallback.

    Tries to load from local storage first. If that fails and HuggingFace
    is configured, attempts to download from HuggingFace Hub.

    Args:
        storage: Local storage instance for caching

    Returns:
        LSTMArtifacts ready for inference

    Raises:
        HTTPException 503: if no model is available from any source
    """
    # Try local storage first
    try:
        return storage.load_current_artifacts()
    except (ValueError, FileNotFoundError) as local_error:
        logger.info(f"Local model not found: {local_error}")

    # Try HuggingFace if configured
    storage_backend = get_storage_backend()
    hf_model_repo = get_hf_model_repo()

    if storage_backend == "hf" or hf_model_repo:
        if hf_model_repo:
            try:
                from brain_api.storage.huggingface import HuggingFaceModelStorage

                logger.info(f"Attempting to load model from HuggingFace: {hf_model_repo}")
                hf_storage = HuggingFaceModelStorage(
                    repo_id=hf_model_repo,
                    local_cache=storage,
                )
                return hf_storage.download_model(use_cache=True)
            except Exception as hf_error:
                logger.error(f"Failed to load model from HuggingFace: {hf_error}")
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"No model available. Local: model not trained. "
                        f"HuggingFace ({hf_model_repo}): {hf_error}"
                    ),
                ) from None

    # No model available from any source
    raise HTTPException(
        status_code=503,
        detail="No trained model available. Train a model first with POST /train/lstm",
    ) from None


def _sort_predictions(predictions: list[LSTMSymbolPrediction]) -> list[LSTMSymbolPrediction]:
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


def _sort_patchtst_predictions(
    predictions: list[PatchTSTSymbolPrediction],
) -> list[PatchTSTSymbolPrediction]:
    """Sort PatchTST predictions by predicted_weekly_return_pct descending."""
    valid = [p for p in predictions if p.predicted_weekly_return_pct is not None]
    invalid = [p for p in predictions if p.predicted_weekly_return_pct is None]

    valid_sorted = sorted(
        valid,
        key=lambda p: p.predicted_weekly_return_pct,  # type: ignore[arg-type]
        reverse=True,
    )

    return valid_sorted + invalid


# ============================================================================
# PatchTST Inference Endpoint
# ============================================================================


def get_patchtst_storage() -> PatchTSTModelStorage:
    """Get the PatchTST model storage instance."""
    return PatchTSTModelStorage()


def get_patchtst_as_of_date(request: PatchTSTInferenceRequest) -> date:
    """Get the as-of date from request or default to today."""
    if request.as_of_date:
        return date.fromisoformat(request.as_of_date)
    return date.today()


def get_sentiment_parquet_path() -> Path:
    """Get the path to the historical sentiment parquet file."""
    project_root = Path(__file__).parent.parent.parent.parent
    return project_root / "data" / "output" / "daily_sentiment.parquet"


@router.post("/patchtst", response_model=PatchTSTInferenceResponse)
def infer_patchtst(
    request: PatchTSTInferenceRequest,
    storage: PatchTSTModelStorage = Depends(get_patchtst_storage),
) -> PatchTSTInferenceResponse:
    """Predict weekly returns using multi-signal PatchTST model.

    This endpoint uses OHLCV + news sentiment + fundamentals to predict
    weekly returns. It provides richer predictions than the pure-price LSTM.

    Input channels (11 total):
    - OHLCV log returns (5)
    - News sentiment (1)
    - Fundamentals (5): gross_margin, operating_margin, net_margin,
      current_ratio, debt_to_equity

    Args:
        request: PatchTSTInferenceRequest with symbols list

    Returns:
        PatchTSTInferenceResponse with per-symbol predictions and metadata

    Raises:
        HTTPException 503: if no trained model is available
    """
    # Get as-of date
    as_of = get_patchtst_as_of_date(request)

    # Compute holiday-aware week boundaries
    week_boundaries = patchtst_compute_week_boundaries(as_of)

    # Load current model artifacts
    artifacts = _load_patchtst_model_artifacts(storage)

    # Calculate data fetch window
    config = artifacts.config
    from datetime import timedelta

    buffer_days = config.context_length * 2 + 30
    data_start = week_boundaries.target_week_start - timedelta(days=buffer_days)
    data_end = week_boundaries.target_week_start - timedelta(days=1)

    # Fetch price data for all symbols
    logger.info(f"[PatchTST] Fetching prices for {len(request.symbols)} symbols...")
    prices = patchtst_load_prices(request.symbols, data_start, data_end)

    # Fetch news sentiment
    logger.info("[PatchTST] Loading news sentiment...")
    news_sentiment = load_historical_news_sentiment(
        request.symbols, data_start, data_end
    )

    # Fetch fundamentals (from cache)
    logger.info("[PatchTST] Loading fundamentals...")
    fundamentals = load_historical_fundamentals(
        request.symbols, data_start, data_end
    )

    # Build features for each symbol
    features_list = []
    for symbol in request.symbols:
        prices_df = prices.get(symbol)
        news_df = news_sentiment.get(symbol)
        fund_df = fundamentals.get(symbol)

        if prices_df is None or prices_df.empty:
            features_list.append(
                PatchTSTInferenceFeatures(
                    symbol=symbol,
                    features=None,
                    has_enough_history=False,
                    history_days_used=0,
                    data_end_date=None,
                    has_news_data=news_df is not None and len(news_df) > 0,
                    has_fundamentals_data=fund_df is not None and len(fund_df) > 0,
                )
            )
        else:
            features = patchtst_build_inference_features(
                symbol=symbol,
                prices_df=prices_df,
                news_df=news_df,
                fundamentals_df=fund_df,
                config=config,
                cutoff_date=week_boundaries.target_week_start,
            )
            features_list.append(features)

    # Run inference
    predictions = patchtst_run_inference(
        model=artifacts.model,
        feature_scaler=artifacts.feature_scaler,
        features_list=features_list,
        week_boundaries=week_boundaries,
    )

    # Sort predictions
    predictions = _sort_patchtst_predictions(predictions)

    # Determine which signals were available
    signals_used = ["ohlcv"]
    if any(f.has_news_data for f in features_list):
        signals_used.append("news_sentiment")
    if any(f.has_fundamentals_data for f in features_list):
        signals_used.append("fundamentals")

    return PatchTSTInferenceResponse(
        predictions=predictions,
        model_version=artifacts.version,
        as_of_date=as_of.isoformat(),
        target_week_start=week_boundaries.target_week_start.isoformat(),
        target_week_end=week_boundaries.target_week_end.isoformat(),
        signals_used=signals_used,
    )


def _load_patchtst_model_artifacts(storage: PatchTSTModelStorage) -> PatchTSTArtifacts:
    """Load PatchTST model artifacts.

    Args:
        storage: PatchTST storage instance

    Returns:
        PatchTSTArtifacts ready for inference

    Raises:
        HTTPException 503: if no model is available
    """
    try:
        return storage.load_current_artifacts()
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"PatchTST model not found: {e}")
        raise HTTPException(
            status_code=503,
            detail="No trained PatchTST model available. Train a model first with POST /train/patchtst",
        ) from None


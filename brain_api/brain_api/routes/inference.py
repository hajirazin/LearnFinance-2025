"""Inference endpoints for ML models."""

import logging
from datetime import date
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from brain_api.core.config import get_hf_lstm_model_repo, get_storage_backend
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
    PPOLSTMArtifacts,
    PPOLSTMLocalStorage,
    PPOPatchTSTArtifacts,
    PPOPatchTSTLocalStorage,
    SACLSTMArtifacts,
    SACLSTMLocalStorage,
    SACPatchTSTArtifacts,
    SACPatchTSTLocalStorage,
)

# Import PPO inference components
from brain_api.core.ppo_lstm import run_ppo_inference
from brain_api.core.ppo_patchtst import run_ppo_patchtst_inference

# Import SAC inference components
from brain_api.core.sac_lstm import run_sac_inference
from brain_api.core.sac_patchtst import run_sac_inference as run_sac_patchtst_inference

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
    import time

    t_start = time.time()
    logger.info(f"[LSTM] Starting inference for {len(request.symbols)} symbols")
    logger.info(f"[LSTM] Symbols: {request.symbols}")

    # Get as-of date
    as_of = get_as_of_date(request)
    logger.info(f"[LSTM] As-of date: {as_of}")

    # Compute holiday-aware week boundaries
    week_boundaries = week_boundary_computer(as_of)
    logger.info(f"[LSTM] Target week: {week_boundaries.target_week_start} to {week_boundaries.target_week_end}")

    # Load current model artifacts (try local first, then HuggingFace)
    logger.info("[LSTM] Loading model artifacts...")
    t0 = time.time()
    artifacts = _load_model_artifacts(storage)
    t_model = time.time() - t0
    logger.info(f"[LSTM] Model loaded in {t_model:.2f}s: version={artifacts.version}")

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
    logger.info(f"[LSTM] Fetching prices for {len(request.symbols)} symbols ({data_start} to {data_end})...")
    t0 = time.time()
    prices = price_loader(request.symbols, data_start, data_end)
    t_prices = time.time() - t0
    logger.info(f"[LSTM] Loaded prices for {len(prices)}/{len(request.symbols)} symbols in {t_prices:.1f}s")

    # Build features for each symbol
    logger.info("[LSTM] Building feature sequences...")
    t0 = time.time()
    features_list = []
    symbols_with_data = 0
    symbols_missing_data = []
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
            symbols_missing_data.append(symbol)
        else:
            features = build_inference_features(
                symbol=symbol,
                prices_df=prices_df,
                config=config,
                cutoff_date=week_boundaries.target_week_start,
            )
            features_list.append(features)
            if features.has_enough_history:
                symbols_with_data += 1
            else:
                symbols_missing_data.append(symbol)
    t_features = time.time() - t0
    logger.info(f"[LSTM] Features built in {t_features:.2f}s: {symbols_with_data} symbols ready")
    if symbols_missing_data:
        logger.warning(f"[LSTM] Symbols with insufficient data: {symbols_missing_data}")

    # Run inference
    logger.info("[LSTM] Running model inference...")
    t0 = time.time()
    predictions = run_inference(
        model=artifacts.model,
        feature_scaler=artifacts.feature_scaler,
        features_list=features_list,
        week_boundaries=week_boundaries,
    )
    t_infer = time.time() - t0
    logger.info(f"[LSTM] Inference complete in {t_infer:.2f}s")

    # Sort predictions: highest gain → highest loss, with null/insufficient-history at the end
    predictions = _sort_predictions(predictions)

    # Summary
    valid_predictions = [p for p in predictions if p.predicted_weekly_return_pct is not None]
    t_total = time.time() - t_start
    logger.info(f"[LSTM] Request complete: {len(valid_predictions)}/{len(request.symbols)} predictions in {t_total:.2f}s")
    if valid_predictions:
        top = valid_predictions[0]
        bottom = valid_predictions[-1]
        logger.info(f"[LSTM] Top: {top.symbol} ({top.predicted_weekly_return_pct:+.2f}%), Bottom: {bottom.symbol} ({bottom.predicted_weekly_return_pct:+.2f}%)")

    return LSTMInferenceResponse(
        predictions=predictions,
        model_version=artifacts.version,
        as_of_date=as_of.isoformat(),
        target_week_start=week_boundaries.target_week_start.isoformat(),
        target_week_end=week_boundaries.target_week_end.isoformat(),
    )


def _load_model_artifacts_generic(
    model_type: str,
    local_storage: LocalModelStorage | PatchTSTModelStorage,
    hf_storage_class: type,
) -> LSTMArtifacts | PatchTSTArtifacts:
    """Load model artifacts with HuggingFace fallback.

    Generic helper that handles local → HuggingFace fallback for any model type.

    Args:
        model_type: Model type identifier (e.g., "LSTM", "PatchTST")
        local_storage: Local storage instance for caching
        hf_storage_class: HuggingFace storage class to use for fallback

    Returns:
        Model artifacts ready for inference

    Raises:
        HTTPException 503: if no model is available from any source
    """
    # Try local storage first
    try:
        return local_storage.load_current_artifacts()
    except (ValueError, FileNotFoundError) as local_error:
        logger.info(f"[{model_type}] Local model not found: {local_error}")

    # Try HuggingFace if configured
    storage_backend = get_storage_backend()
    hf_model_repo = get_hf_lstm_model_repo()

    if storage_backend == "hf" or hf_model_repo:
        if hf_model_repo:
            try:
                logger.info(f"[{model_type}] Attempting to load model from HuggingFace: {hf_model_repo}")
                hf_storage = hf_storage_class(
                    repo_id=hf_model_repo,
                    local_cache=local_storage,
                )
                return hf_storage.download_model(use_cache=True)
            except Exception as hf_error:
                logger.error(f"[{model_type}] Failed to load model from HuggingFace: {hf_error}")
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"No {model_type} model available. Local: model not trained. "
                        f"HuggingFace ({hf_model_repo}): {hf_error}"
                    ),
                ) from None

    # No model available from any source
    raise HTTPException(
        status_code=503,
        detail=f"No trained {model_type} model available. Train a model first with POST /train/{model_type.lower()}",
    ) from None


def _load_model_artifacts(storage: LocalModelStorage) -> LSTMArtifacts:
    """Load LSTM model artifacts with HuggingFace fallback."""
    from brain_api.storage.huggingface import HuggingFaceModelStorage

    return _load_model_artifacts_generic("LSTM", storage, HuggingFaceModelStorage)


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
    import time

    t_start = time.time()
    logger.info(f"[PatchTST] Starting inference for {len(request.symbols)} symbols")
    logger.info(f"[PatchTST] Symbols: {request.symbols}")

    # Get as-of date
    as_of = get_patchtst_as_of_date(request)
    logger.info(f"[PatchTST] As-of date: {as_of}")

    # Compute holiday-aware week boundaries
    week_boundaries = patchtst_compute_week_boundaries(as_of)
    logger.info(f"[PatchTST] Target week: {week_boundaries.target_week_start} to {week_boundaries.target_week_end}")

    # Load current model artifacts
    logger.info("[PatchTST] Loading model artifacts...")
    t0 = time.time()
    artifacts = _load_patchtst_model_artifacts(storage)
    t_model = time.time() - t0
    logger.info(f"[PatchTST] Model loaded in {t_model:.2f}s: version={artifacts.version}")

    # Calculate data fetch window
    config = artifacts.config
    from datetime import timedelta

    buffer_days = config.context_length * 2 + 30
    data_start = week_boundaries.target_week_start - timedelta(days=buffer_days)
    data_end = week_boundaries.target_week_start - timedelta(days=1)
    logger.info(f"[PatchTST] Data window: {data_start} to {data_end}")

    # Fetch price data for all symbols
    logger.info(f"[PatchTST] Fetching prices for {len(request.symbols)} symbols...")
    t0 = time.time()
    prices = patchtst_load_prices(request.symbols, data_start, data_end)
    t_prices = time.time() - t0
    logger.info(f"[PatchTST] Loaded prices for {len(prices)}/{len(request.symbols)} symbols in {t_prices:.1f}s")

    # Fetch news sentiment
    logger.info("[PatchTST] Loading news sentiment...")
    t0 = time.time()
    news_sentiment = load_historical_news_sentiment(
        request.symbols, data_start, data_end
    )
    t_news = time.time() - t0
    logger.info(f"[PatchTST] Loaded news for {len(news_sentiment)}/{len(request.symbols)} symbols in {t_news:.1f}s")

    # Fetch fundamentals (from cache)
    logger.info("[PatchTST] Loading fundamentals...")
    t0 = time.time()
    fundamentals = load_historical_fundamentals(
        request.symbols, data_start, data_end
    )
    t_fund = time.time() - t0
    logger.info(f"[PatchTST] Loaded fundamentals for {len(fundamentals)}/{len(request.symbols)} symbols in {t_fund:.1f}s")

    # Build features for each symbol
    logger.info("[PatchTST] Building feature sequences...")
    t0 = time.time()
    features_list = []
    symbols_with_data = 0
    symbols_missing_data = []
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
            symbols_missing_data.append(symbol)
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
            if features.has_enough_history:
                symbols_with_data += 1
            else:
                symbols_missing_data.append(symbol)
    t_features = time.time() - t0
    logger.info(f"[PatchTST] Features built in {t_features:.2f}s: {symbols_with_data} symbols ready")
    if symbols_missing_data:
        logger.warning(f"[PatchTST] Symbols with insufficient data: {symbols_missing_data}")

    # Run inference
    logger.info("[PatchTST] Running model inference...")
    t0 = time.time()
    predictions = patchtst_run_inference(
        model=artifacts.model,
        feature_scaler=artifacts.feature_scaler,
        features_list=features_list,
        week_boundaries=week_boundaries,
        config=config,
    )
    t_infer = time.time() - t0
    logger.info(f"[PatchTST] Inference complete in {t_infer:.2f}s")

    # Sort predictions
    predictions = _sort_patchtst_predictions(predictions)

    # Determine which signals were available
    signals_used = ["ohlcv"]
    if any(f.has_news_data for f in features_list):
        signals_used.append("news_sentiment")
    if any(f.has_fundamentals_data for f in features_list):
        signals_used.append("fundamentals")

    # Summary
    valid_predictions = [p for p in predictions if p.predicted_weekly_return_pct is not None]
    t_total = time.time() - t_start
    logger.info(f"[PatchTST] Request complete: {len(valid_predictions)}/{len(request.symbols)} predictions in {t_total:.2f}s")
    logger.info(f"[PatchTST] Signals used: {signals_used}")
    if valid_predictions:
        top = valid_predictions[0]
        bottom = valid_predictions[-1]
        logger.info(f"[PatchTST] Top: {top.symbol} ({top.predicted_weekly_return_pct:+.2f}%), Bottom: {bottom.symbol} ({bottom.predicted_weekly_return_pct:+.2f}%)")

    return PatchTSTInferenceResponse(
        predictions=predictions,
        model_version=artifacts.version,
        as_of_date=as_of.isoformat(),
        target_week_start=week_boundaries.target_week_start.isoformat(),
        target_week_end=week_boundaries.target_week_end.isoformat(),
        signals_used=signals_used,
    )


def _load_patchtst_model_artifacts(storage: PatchTSTModelStorage) -> PatchTSTArtifacts:
    """Load PatchTST model artifacts with HuggingFace fallback."""
    from brain_api.storage.huggingface import PatchTSTHuggingFaceModelStorage

    return _load_model_artifacts_generic("PatchTST", storage, PatchTSTHuggingFaceModelStorage)


# ============================================================================
# PPO + LSTM Inference Endpoint
# ============================================================================


class Position(BaseModel):
    """A single position in the portfolio."""

    symbol: str
    market_value: float = Field(..., ge=0)


class PortfolioSnapshot(BaseModel):
    """Current portfolio state from Alpaca or similar broker."""

    cash: float = Field(..., ge=0)
    positions: list[Position] = Field(default_factory=list)


class WeightChange(BaseModel):
    """Weight change for a single symbol."""

    symbol: str
    current_weight: float
    target_weight: float
    change: float


class PPOLSTMInferenceRequest(BaseModel):
    """Request model for PPO + LSTM inference endpoint."""

    portfolio: PortfolioSnapshot = Field(
        ...,
        description="Current portfolio state (cash + positions)",
    )
    as_of_date: str | None = Field(
        None,
        description="Reference date for inference (YYYY-MM-DD). Defaults to today.",
    )


class PPOLSTMInferenceResponse(BaseModel):
    """Response model for PPO + LSTM inference endpoint."""

    target_weights: dict[str, float]
    turnover: float
    target_week_start: str  # YYYY-MM-DD
    target_week_end: str  # YYYY-MM-DD
    model_version: str
    weight_changes: list[WeightChange]


def get_ppo_lstm_storage() -> PPOLSTMLocalStorage:
    """Get the PPO + LSTM storage instance."""
    return PPOLSTMLocalStorage()


def get_ppo_lstm_as_of_date(request: PPOLSTMInferenceRequest) -> date:
    """Get the as-of date from request or default to today."""
    if request.as_of_date:
        return date.fromisoformat(request.as_of_date)
    return date.today()


@router.post("/ppo_lstm", response_model=PPOLSTMInferenceResponse)
def infer_ppo_lstm(
    request: PPOLSTMInferenceRequest,
    storage: PPOLSTMLocalStorage = Depends(get_ppo_lstm_storage),
) -> PPOLSTMInferenceResponse:
    """Get target portfolio weights from PPO policy.

    This endpoint:
    1. Loads the current PPO model
    2. Normalizes the portfolio snapshot to weights
    3. Builds state vector with current signals + LSTM forecasts
    4. Runs PPO inference to get target weights
    5. Returns target weights and turnover

    Args:
        request: Portfolio snapshot (cash + positions)

    Returns:
        Target weights and execution metadata
    """
    import time

    t_start = time.time()
    logger.info("[PPO_LSTM] Starting inference")

    # Get as-of date
    as_of = get_ppo_lstm_as_of_date(request)
    logger.info(f"[PPO_LSTM] As-of date: {as_of}")

    # Compute target week boundaries
    week_boundaries = compute_week_boundaries(as_of)
    logger.info(f"[PPO_LSTM] Target week: {week_boundaries.target_week_start} to {week_boundaries.target_week_end}")

    # Load model artifacts
    logger.info("[PPO_LSTM] Loading model artifacts...")
    try:
        artifacts = storage.load_current_artifacts()
    except ValueError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e),
        )

    logger.info(f"[PPO_LSTM] Model loaded: version={artifacts.version}")

    # Convert portfolio snapshot to values dict
    cash_value = request.portfolio.cash
    position_values = {
        pos.symbol: pos.market_value
        for pos in request.portfolio.positions
    }

    # Build placeholder signals (simplified - would fetch real data in production)
    signals = {}
    for symbol in artifacts.symbol_order:
        signals[symbol] = {
            "news_sentiment": 0.0,
            "gross_margin": 0.0,
            "operating_margin": 0.0,
            "net_margin": 0.0,
            "current_ratio": 0.0,
            "debt_to_equity": 0.0,
            "fundamental_age": 0.0,
        }

    # Build placeholder forecast features
    forecast_features = {symbol: 0.0 for symbol in artifacts.symbol_order}

    # Run inference
    logger.info("[PPO_LSTM] Running inference...")
    result = run_ppo_inference(
        model=artifacts.model,
        scaler=artifacts.scaler,
        config=artifacts.config,
        symbol_order=artifacts.symbol_order,
        signals=signals,
        forecast_features=forecast_features,
        cash_value=cash_value,
        position_values=position_values,
        target_week_start=week_boundaries.target_week_start,
        target_week_end=week_boundaries.target_week_end,
        model_version=artifacts.version,
    )

    # Build weight changes list
    weight_changes = []
    for symbol in artifacts.symbol_order:
        weight_changes.append(WeightChange(
            symbol=symbol,
            current_weight=result.current_weights.get(symbol, 0.0),
            target_weight=result.target_weights.get(symbol, 0.0),
            change=result.weight_changes.get(symbol, 0.0),
        ))
    # Add CASH
    weight_changes.append(WeightChange(
        symbol="CASH",
        current_weight=result.current_weights.get("CASH", 0.0),
        target_weight=result.target_weights.get("CASH", 0.0),
        change=result.weight_changes.get("CASH", 0.0),
    ))

    t_total = time.time() - t_start
    logger.info(f"[PPO_LSTM] Inference complete in {t_total:.2f}s, turnover={result.turnover:.4f}")

    return PPOLSTMInferenceResponse(
        target_weights=result.target_weights,
        turnover=result.turnover,
        target_week_start=result.target_week_start,
        target_week_end=result.target_week_end,
        model_version=result.model_version,
        weight_changes=weight_changes,
    )


# ============================================================================
# PPO + PatchTST Inference Endpoint
# ============================================================================


class PPOPatchTSTInferenceRequest(BaseModel):
    """Request model for PPO + PatchTST inference endpoint."""

    portfolio: PortfolioSnapshot = Field(
        ...,
        description="Current portfolio state (cash + positions)",
    )
    as_of_date: str | None = Field(
        None,
        description="Reference date for inference (YYYY-MM-DD). Defaults to today.",
    )


class PPOPatchTSTInferenceResponse(BaseModel):
    """Response model for PPO + PatchTST inference endpoint."""

    target_weights: dict[str, float]
    turnover: float
    target_week_start: str  # YYYY-MM-DD
    target_week_end: str  # YYYY-MM-DD
    model_version: str
    weight_changes: list[WeightChange]


def get_ppo_patchtst_storage() -> PPOPatchTSTLocalStorage:
    """Get the PPO + PatchTST storage instance."""
    return PPOPatchTSTLocalStorage()


def get_ppo_patchtst_as_of_date(request: PPOPatchTSTInferenceRequest) -> date:
    """Get the as-of date from request or default to today."""
    if request.as_of_date:
        return date.fromisoformat(request.as_of_date)
    return date.today()


@router.post("/ppo_patchtst", response_model=PPOPatchTSTInferenceResponse)
def infer_ppo_patchtst(
    request: PPOPatchTSTInferenceRequest,
    storage: PPOPatchTSTLocalStorage = Depends(get_ppo_patchtst_storage),
) -> PPOPatchTSTInferenceResponse:
    """Get target portfolio weights from PPO + PatchTST policy.

    This endpoint:
    1. Loads the current PPO + PatchTST model
    2. Normalizes the portfolio snapshot to weights
    3. Builds state vector with current signals + PatchTST forecasts
    4. Runs PPO inference to get target weights
    5. Returns target weights and turnover

    Args:
        request: Portfolio snapshot (cash + positions)

    Returns:
        Target weights and execution metadata
    """
    import time

    t_start = time.time()
    logger.info("[PPO_PatchTST] Starting inference")

    # Get as-of date
    as_of = get_ppo_patchtst_as_of_date(request)
    logger.info(f"[PPO_PatchTST] As-of date: {as_of}")

    # Compute target week boundaries
    week_boundaries = compute_week_boundaries(as_of)
    logger.info(f"[PPO_PatchTST] Target week: {week_boundaries.target_week_start} to {week_boundaries.target_week_end}")

    # Load model artifacts
    logger.info("[PPO_PatchTST] Loading model artifacts...")
    try:
        artifacts = storage.load_current_artifacts()
    except ValueError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e),
        )

    logger.info(f"[PPO_PatchTST] Model loaded: version={artifacts.version}")

    # Convert portfolio snapshot to values dict
    cash_value = request.portfolio.cash
    position_values = {
        pos.symbol: pos.market_value
        for pos in request.portfolio.positions
    }

    # Build placeholder signals
    signals = {}
    for symbol in artifacts.symbol_order:
        signals[symbol] = {
            "news_sentiment": 0.0,
            "gross_margin": 0.0,
            "operating_margin": 0.0,
            "net_margin": 0.0,
            "current_ratio": 0.0,
            "debt_to_equity": 0.0,
            "fundamental_age": 0.0,
        }

    # Build placeholder forecast features
    forecast_features = {symbol: 0.0 for symbol in artifacts.symbol_order}

    # Run inference
    logger.info("[PPO_PatchTST] Running inference...")
    result = run_ppo_patchtst_inference(
        model=artifacts.model,
        scaler=artifacts.scaler,
        config=artifacts.config,
        symbol_order=artifacts.symbol_order,
        signals=signals,
        forecast_features=forecast_features,
        cash_value=cash_value,
        position_values=position_values,
        target_week_start=week_boundaries.target_week_start,
        target_week_end=week_boundaries.target_week_end,
        model_version=artifacts.version,
    )

    # Build weight changes list
    weight_changes = []
    for symbol in artifacts.symbol_order:
        weight_changes.append(WeightChange(
            symbol=symbol,
            current_weight=result.current_weights.get(symbol, 0.0),
            target_weight=result.target_weights.get(symbol, 0.0),
            change=result.weight_changes.get(symbol, 0.0),
        ))
    # Add CASH
    weight_changes.append(WeightChange(
        symbol="CASH",
        current_weight=result.current_weights.get("CASH", 0.0),
        target_weight=result.target_weights.get("CASH", 0.0),
        change=result.weight_changes.get("CASH", 0.0),
    ))

    t_total = time.time() - t_start
    logger.info(f"[PPO_PatchTST] Inference complete in {t_total:.2f}s, turnover={result.turnover:.4f}")

    return PPOPatchTSTInferenceResponse(
        target_weights=result.target_weights,
        turnover=result.turnover,
        target_week_start=result.target_week_start,
        target_week_end=result.target_week_end,
        model_version=result.model_version,
        weight_changes=weight_changes,
    )


# ============================================================================
# SAC + LSTM Inference Endpoint
# ============================================================================


class SACLSTMInferenceRequest(BaseModel):
    """Request model for SAC + LSTM inference endpoint."""

    portfolio: PortfolioSnapshot = Field(
        ...,
        description="Current portfolio state (cash + positions)",
    )
    as_of_date: str | None = Field(
        None,
        description="Reference date for inference (YYYY-MM-DD). Defaults to today.",
    )


class SACLSTMInferenceResponse(BaseModel):
    """Response model for SAC + LSTM inference endpoint."""

    target_weights: dict[str, float]
    turnover: float
    target_week_start: str  # YYYY-MM-DD
    target_week_end: str  # YYYY-MM-DD
    model_version: str
    weight_changes: list[WeightChange]


def get_sac_lstm_storage() -> SACLSTMLocalStorage:
    """Get the SAC + LSTM storage instance."""
    return SACLSTMLocalStorage()


def get_sac_lstm_as_of_date(request: SACLSTMInferenceRequest) -> date:
    """Get the as-of date from request or default to today."""
    if request.as_of_date:
        return date.fromisoformat(request.as_of_date)
    return date.today()


@router.post("/sac_lstm", response_model=SACLSTMInferenceResponse)
def infer_sac_lstm(
    request: SACLSTMInferenceRequest,
    storage: SACLSTMLocalStorage = Depends(get_sac_lstm_storage),
) -> SACLSTMInferenceResponse:
    """Get target portfolio weights from SAC + LSTM policy.

    This endpoint:
    1. Loads the current SAC model
    2. Normalizes the portfolio snapshot to weights
    3. Builds state vector with current signals + LSTM forecasts
    4. Runs SAC inference to get target weights
    5. Returns target weights and turnover

    Args:
        request: Portfolio snapshot (cash + positions)

    Returns:
        Target weights and execution metadata
    """
    import time
    import numpy as np

    t_start = time.time()
    logger.info("[SAC_LSTM] Starting inference")

    # Get as-of date
    as_of = get_sac_lstm_as_of_date(request)
    logger.info(f"[SAC_LSTM] As-of date: {as_of}")

    # Compute target week boundaries
    week_boundaries = compute_week_boundaries(as_of)
    logger.info(f"[SAC_LSTM] Target week: {week_boundaries.target_week_start} to {week_boundaries.target_week_end}")

    # Load model artifacts
    logger.info("[SAC_LSTM] Loading model artifacts...")
    try:
        artifacts = storage.load_current_artifacts()
    except ValueError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e),
        )

    logger.info(f"[SAC_LSTM] Model loaded: version={artifacts.version}")

    # Convert portfolio snapshot to values dict
    cash_value = request.portfolio.cash
    position_values = {
        pos.symbol: pos.market_value
        for pos in request.portfolio.positions
    }

    # Compute total portfolio value
    total_value = cash_value + sum(position_values.values())

    # Build current weights vector (including CASH)
    n_stocks = len(artifacts.symbol_order)
    current_weights = np.zeros(n_stocks + 1)
    for i, symbol in enumerate(artifacts.symbol_order):
        if symbol in position_values and total_value > 0:
            current_weights[i] = position_values[symbol] / total_value
    current_weights[-1] = cash_value / total_value if total_value > 0 else 1.0

    # Build placeholder signals (simplified - would fetch real data in production)
    signals = {}
    for symbol in artifacts.symbol_order:
        signals[symbol] = {
            "news_sentiment": 0.0,
            "gross_margin": 0.0,
            "operating_margin": 0.0,
            "net_margin": 0.0,
            "current_ratio": 0.0,
            "debt_to_equity": 0.0,
            "fundamental_age": 0.0,
        }

    # Build placeholder forecast features
    forecast_features = {symbol: 0.0 for symbol in artifacts.symbol_order}

    # Run inference
    logger.info("[SAC_LSTM] Running inference...")
    result = run_sac_inference(
        actor=artifacts.actor,
        scaler=artifacts.scaler,
        config=artifacts.config,
        symbol_order=artifacts.symbol_order,
        current_weights=current_weights,
        signals=signals,
        forecast_features=forecast_features,
        model_version=artifacts.version,
    )

    # Build weight changes list
    weight_changes = []
    for symbol in artifacts.symbol_order:
        current_w = current_weights[artifacts.symbol_order.index(symbol)]
        target_w = result.allocation.get(symbol, 0.0)
        weight_changes.append(WeightChange(
            symbol=symbol,
            current_weight=current_w,
            target_weight=target_w,
            change=target_w - current_w,
        ))
    # Add CASH
    weight_changes.append(WeightChange(
        symbol="CASH",
        current_weight=current_weights[-1],
        target_weight=result.allocation.get("CASH", 0.0),
        change=result.allocation.get("CASH", 0.0) - current_weights[-1],
    ))

    t_total = time.time() - t_start
    logger.info(f"[SAC_LSTM] Inference complete in {t_total:.2f}s, turnover={result.turnover:.4f}")

    return SACLSTMInferenceResponse(
        target_weights=result.allocation,
        turnover=result.turnover,
        target_week_start=week_boundaries.target_week_start.isoformat(),
        target_week_end=week_boundaries.target_week_end.isoformat(),
        model_version=result.model_version,
        weight_changes=weight_changes,
    )


# ============================================================================
# SAC + PatchTST Inference Endpoint
# ============================================================================


class SACPatchTSTInferenceRequest(BaseModel):
    """Request model for SAC + PatchTST inference endpoint."""

    portfolio: PortfolioSnapshot = Field(
        ...,
        description="Current portfolio state (cash + positions)",
    )
    as_of_date: str | None = Field(
        None,
        description="Reference date for inference (YYYY-MM-DD). Defaults to today.",
    )


class SACPatchTSTInferenceResponse(BaseModel):
    """Response model for SAC + PatchTST inference endpoint."""

    target_weights: dict[str, float]
    turnover: float
    target_week_start: str  # YYYY-MM-DD
    target_week_end: str  # YYYY-MM-DD
    model_version: str
    weight_changes: list[WeightChange]


def get_sac_patchtst_storage() -> SACPatchTSTLocalStorage:
    """Get the SAC + PatchTST storage instance."""
    return SACPatchTSTLocalStorage()


def get_sac_patchtst_as_of_date(request: SACPatchTSTInferenceRequest) -> date:
    """Get the as-of date from request or default to today."""
    if request.as_of_date:
        return date.fromisoformat(request.as_of_date)
    return date.today()


@router.post("/sac_patchtst", response_model=SACPatchTSTInferenceResponse)
def infer_sac_patchtst(
    request: SACPatchTSTInferenceRequest,
    storage: SACPatchTSTLocalStorage = Depends(get_sac_patchtst_storage),
) -> SACPatchTSTInferenceResponse:
    """Get target portfolio weights from SAC + PatchTST policy.

    This endpoint:
    1. Loads the current SAC + PatchTST model
    2. Normalizes the portfolio snapshot to weights
    3. Builds state vector with current signals + PatchTST forecasts
    4. Runs SAC inference to get target weights
    5. Returns target weights and turnover

    Args:
        request: Portfolio snapshot (cash + positions)

    Returns:
        Target weights and execution metadata
    """
    import time
    import numpy as np

    t_start = time.time()
    logger.info("[SAC_PatchTST] Starting inference")

    # Get as-of date
    as_of = get_sac_patchtst_as_of_date(request)
    logger.info(f"[SAC_PatchTST] As-of date: {as_of}")

    # Compute target week boundaries
    week_boundaries = compute_week_boundaries(as_of)
    logger.info(f"[SAC_PatchTST] Target week: {week_boundaries.target_week_start} to {week_boundaries.target_week_end}")

    # Load model artifacts
    logger.info("[SAC_PatchTST] Loading model artifacts...")
    try:
        artifacts = storage.load_current_artifacts()
    except ValueError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e),
        )

    logger.info(f"[SAC_PatchTST] Model loaded: version={artifacts.version}")

    # Convert portfolio snapshot to values dict
    cash_value = request.portfolio.cash
    position_values = {
        pos.symbol: pos.market_value
        for pos in request.portfolio.positions
    }

    # Compute total portfolio value
    total_value = cash_value + sum(position_values.values())

    # Build current weights vector (including CASH)
    n_stocks = len(artifacts.symbol_order)
    current_weights = np.zeros(n_stocks + 1)
    for i, symbol in enumerate(artifacts.symbol_order):
        if symbol in position_values and total_value > 0:
            current_weights[i] = position_values[symbol] / total_value
    current_weights[-1] = cash_value / total_value if total_value > 0 else 1.0

    # Build placeholder signals
    signals = {}
    for symbol in artifacts.symbol_order:
        signals[symbol] = {
            "news_sentiment": 0.0,
            "gross_margin": 0.0,
            "operating_margin": 0.0,
            "net_margin": 0.0,
            "current_ratio": 0.0,
            "debt_to_equity": 0.0,
            "fundamental_age": 0.0,
        }

    # Build placeholder forecast features
    forecast_features = {symbol: 0.0 for symbol in artifacts.symbol_order}

    # Run inference
    logger.info("[SAC_PatchTST] Running inference...")
    result = run_sac_patchtst_inference(
        actor=artifacts.actor,
        scaler=artifacts.scaler,
        config=artifacts.config,
        symbol_order=artifacts.symbol_order,
        current_weights=current_weights,
        signals=signals,
        forecast_features=forecast_features,
        model_version=artifacts.version,
    )

    # Build weight changes list
    weight_changes = []
    for symbol in artifacts.symbol_order:
        current_w = current_weights[artifacts.symbol_order.index(symbol)]
        target_w = result.allocation.get(symbol, 0.0)
        weight_changes.append(WeightChange(
            symbol=symbol,
            current_weight=current_w,
            target_weight=target_w,
            change=target_w - current_w,
        ))
    # Add CASH
    weight_changes.append(WeightChange(
        symbol="CASH",
        current_weight=current_weights[-1],
        target_weight=result.allocation.get("CASH", 0.0),
        change=result.allocation.get("CASH", 0.0) - current_weights[-1],
    ))

    t_total = time.time() - t_start
    logger.info(f"[SAC_PatchTST] Inference complete in {t_total:.2f}s, turnover={result.turnover:.4f}")

    return SACPatchTSTInferenceResponse(
        target_weights=result.allocation,
        turnover=result.turnover,
        target_week_start=week_boundaries.target_week_start.isoformat(),
        target_week_end=week_boundaries.target_week_end.isoformat(),
        model_version=result.model_version,
        weight_changes=weight_changes,
    )

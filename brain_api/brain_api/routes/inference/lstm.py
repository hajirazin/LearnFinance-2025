"""LSTM inference endpoint."""

import logging
import time
from datetime import timedelta

from fastapi import APIRouter, Depends

from brain_api.core.lstm import (
    InferenceFeatures,
    build_inference_features,
    run_inference,
)
from brain_api.storage.local import LocalModelStorage

from .dependencies import (
    PriceLoader,
    WeekBoundaryComputer,
    get_as_of_date,
    get_price_loader,
    get_storage,
    get_week_boundary_computer,
)
from .helpers import _load_model_artifacts, _sort_predictions
from .models import LSTMInferenceRequest, LSTMInferenceResponse

router = APIRouter()
logger = logging.getLogger(__name__)


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


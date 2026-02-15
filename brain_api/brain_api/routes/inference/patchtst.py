"""PatchTST inference endpoint."""

import logging
import time
from datetime import timedelta

from fastapi import APIRouter, Depends

from brain_api.core.inference_utils import compute_week_from_cutoff
from brain_api.core.patchtst import (
    InferenceFeatures as PatchTSTInferenceFeatures,
)
from brain_api.core.patchtst import (
    build_inference_features as patchtst_build_inference_features,
)
from brain_api.core.patchtst import (
    load_prices_yfinance as patchtst_load_prices,
)
from brain_api.core.patchtst import (
    run_inference as patchtst_run_inference,
)
from brain_api.core.realtime_signals import RealTimeSignalBuilder
from brain_api.storage.local import PatchTSTModelStorage

from .dependencies import get_patchtst_as_of_date, get_patchtst_storage
from .helpers import _load_patchtst_model_artifacts, _sort_patchtst_predictions
from .models import PatchTSTInferenceRequest, PatchTSTInferenceResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/patchtst", response_model=PatchTSTInferenceResponse)
def infer_patchtst(
    request: PatchTSTInferenceRequest,
    storage: PatchTSTModelStorage = Depends(get_patchtst_storage),
) -> PatchTSTInferenceResponse:
    """Predict weekly returns using OHLCV PatchTST model.

    This endpoint uses 5-channel OHLCV log returns to predict weekly returns.
    Channel-independent shared Transformer weights learn temporal patterns
    from all 5 related OHLCV signals.

    Input channels (5 total):
    - OHLCV log returns: open_ret, high_ret, low_ret, close_ret, volume_ret

    Args:
        request: PatchTSTInferenceRequest with symbols list

    Returns:
        PatchTSTInferenceResponse with per-symbol predictions and metadata

    Raises:
        HTTPException 503: if no trained model is available
    """
    t_start = time.time()
    logger.info(f"[PatchTST] Starting inference for {len(request.symbols)} symbols")
    logger.info(f"[PatchTST] Symbols: {request.symbols}")

    # Get cutoff date (always a Friday)
    cutoff_date = get_patchtst_as_of_date(request)
    logger.info(f"[PatchTST] Cutoff date: {cutoff_date}")

    # Compute holiday-aware week boundaries for the week AFTER cutoff
    week_boundaries = compute_week_from_cutoff(cutoff_date)
    logger.info(
        f"[PatchTST] Target week: {week_boundaries.target_week_start} to {week_boundaries.target_week_end}"
    )

    # Load current model artifacts
    logger.info("[PatchTST] Loading model artifacts...")
    t0 = time.time()
    artifacts = _load_patchtst_model_artifacts(storage)
    t_model = time.time() - t0
    logger.info(
        f"[PatchTST] Model loaded in {t_model:.2f}s: version={artifacts.version}"
    )

    # Calculate data fetch window
    config = artifacts.config
    buffer_days = config.context_length * 2 + 30
    data_start = week_boundaries.target_week_start - timedelta(days=buffer_days)
    data_end = week_boundaries.target_week_start - timedelta(days=1)
    logger.info(f"[PatchTST] Data window: {data_start} to {data_end}")

    # Fetch price data for all symbols
    logger.info(f"[PatchTST] Fetching prices for {len(request.symbols)} symbols...")
    t0 = time.time()
    prices = patchtst_load_prices(request.symbols, data_start, data_end)
    t_prices = time.time() - t0
    logger.info(
        f"[PatchTST] Loaded prices for {len(prices)}/{len(request.symbols)} symbols in {t_prices:.1f}s"
    )

    # Fetch news sentiment (real-time from yfinance + FinBERT)
    logger.info("[PatchTST] Fetching real-time news sentiment...")
    t0 = time.time()
    signal_builder = RealTimeSignalBuilder()
    news_sentiment = signal_builder.build_news_dataframes(
        request.symbols, data_start, data_end
    )
    t_news = time.time() - t0
    logger.info(
        f"[PatchTST] Fetched news for {len(news_sentiment)}/{len(request.symbols)} symbols in {t_news:.1f}s"
    )

    # Fetch fundamentals (real-time from yfinance)
    logger.info("[PatchTST] Fetching real-time fundamentals...")
    t0 = time.time()
    fundamentals = signal_builder.build_fundamentals_dataframes(
        request.symbols, data_start, data_end
    )
    t_fund = time.time() - t0
    logger.info(
        f"[PatchTST] Fetched fundamentals for {len(fundamentals)}/{len(request.symbols)} symbols in {t_fund:.1f}s"
    )

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
    logger.info(
        f"[PatchTST] Features built in {t_features:.2f}s: {symbols_with_data} symbols ready"
    )
    if symbols_missing_data:
        logger.warning(
            f"[PatchTST] Symbols with insufficient data: {symbols_missing_data}"
        )

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

    # Model only uses OHLCV channels (news/fundamentals loaded for metadata flags only)
    signals_used = ["ohlcv"]

    # Summary
    valid_predictions = [
        p for p in predictions if p.predicted_weekly_return_pct is not None
    ]
    t_total = time.time() - t_start
    logger.info(
        f"[PatchTST] Request complete: {len(valid_predictions)}/{len(request.symbols)} predictions in {t_total:.2f}s"
    )
    logger.info(f"[PatchTST] Signals used: {signals_used}")
    if valid_predictions:
        top = valid_predictions[0]
        bottom = valid_predictions[-1]
        logger.info(
            f"[PatchTST] Top: {top.symbol} ({top.predicted_weekly_return_pct:+.2f}%), Bottom: {bottom.symbol} ({bottom.predicted_weekly_return_pct:+.2f}%)"
        )

    return PatchTSTInferenceResponse(
        predictions=predictions,
        model_version=artifacts.version,
        as_of_date=cutoff_date.isoformat(),
        target_week_start=week_boundaries.target_week_start.isoformat(),
        target_week_end=week_boundaries.target_week_end.isoformat(),
        signals_used=signals_used,
    )

"""PatchTST inference helpers.

Close-only direct 5-day inference.

Single forward pass produces (batch, 5, 1) output -- 5 days x 1 close channel.
RevIN automatically denormalizes output to original log-return scale.
Compound the five close log returns for the weekly return. NO inverse-transform
needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from transformers import PatchTSTForPrediction

from brain_api.core.features import compute_ohlcv_log_returns
from brain_api.core.inference_utils import WeekBoundaries, compute_week_from_cutoff
from brain_api.core.model_types import classify_direction
from brain_api.core.patchtst.config import PatchTSTConfig

if TYPE_CHECKING:
    from brain_api.storage.local import PatchTSTModelStorage

logger = logging.getLogger(__name__)


@dataclass
class InferenceFeatures:
    """Close-return features prepared for inference for a single symbol."""

    symbol: str
    features: np.ndarray | None  # Shape: (context_length, 1) -- close_ret, or None
    has_enough_history: bool
    history_days_used: int
    data_end_date: date | None
    starting_price: (
        float | None
    )  # Starting price (last close before target week) for weekly return calculation


@dataclass
class SymbolPrediction:
    """Prediction result for a single symbol."""

    symbol: str
    predicted_weekly_return_pct: float | None
    direction: str  # "UP", "DOWN", or "FLAT"
    has_enough_history: bool
    history_days_used: int
    data_end_date: str | None
    target_week_start: str
    target_week_end: str
    daily_returns: list[float] | None = None  # 5 daily close_ret predictions


def build_inference_features(
    symbol: str,
    prices_df: pd.DataFrame,
    config: PatchTSTConfig,
    cutoff_date: date,
) -> InferenceFeatures:
    """Build close-only feature sequence for inference (no signals).

    Args:
        symbol: Ticker symbol
        prices_df: DataFrame with OHLCV columns and DatetimeIndex
        config: PatchTST config with context_length and feature settings
        cutoff_date: Features end before this date (typically target_week_start)

    Returns:
        InferenceFeatures with close-return sequence (UNSCALED)
    """
    if prices_df.empty:
        return InferenceFeatures(
            symbol=symbol,
            features=None,
            has_enough_history=False,
            history_days_used=0,
            data_end_date=None,
            starting_price=None,
        )

    if not isinstance(prices_df.index, pd.DatetimeIndex):
        return InferenceFeatures(
            symbol=symbol,
            features=None,
            has_enough_history=False,
            history_days_used=0,
            data_end_date=None,
            starting_price=None,
        )

    # Filter to data before cutoff
    # Handle timezone-aware index by localizing cutoff_ts to match
    cutoff_ts = pd.Timestamp(cutoff_date)
    if prices_df.index.tz is not None:
        cutoff_ts = cutoff_ts.tz_localize(prices_df.index.tz)
    df = prices_df[prices_df.index < cutoff_ts].copy()

    if len(df) < config.context_length + 1:
        return InferenceFeatures(
            symbol=symbol,
            features=None,
            has_enough_history=False,
            history_days_used=len(df),
            data_end_date=df.index[-1].date() if len(df) > 0 else None,
            starting_price=None,
        )

    # Compute OHLCV log returns; select locked close-only channels
    features_df = compute_ohlcv_log_returns(df, use_returns=config.use_returns)

    # Normalize index to timezone-naive for consistent comparisons
    if features_df.index.tz is not None:
        features_df.index = features_df.index.tz_localize(None)

    if len(features_df) < config.context_length:
        return InferenceFeatures(
            symbol=symbol,
            features=None,
            has_enough_history=False,
            history_days_used=len(features_df),
            data_end_date=features_df.index[-1].date()
            if len(features_df) > 0
            else None,
            starting_price=None,
        )

    sequence = (
        features_df[list(config.feature_names)].iloc[-config.context_length :].values
    )  # (context_length, n_channels)
    data_end_date = features_df.index[-1].date()

    # Get starting price: last close price before cutoff_date (for weekly return calculation)
    starting_price = None
    if len(df) > 0:
        try:
            last_close = df.iloc[-1]["close"]
            if pd.notna(last_close) and last_close > 0:
                starting_price = float(last_close)
        except (KeyError, IndexError):
            pass

    return InferenceFeatures(
        symbol=symbol,
        features=sequence,  # (context_length, n_channels) -- UNSCALED
        has_enough_history=True,
        history_days_used=len(features_df),
        data_end_date=data_end_date,
        starting_price=starting_price,
    )


def run_inference(
    model: PatchTSTForPrediction,
    feature_scaler: StandardScaler,
    features_list: list[InferenceFeatures],
    week_boundaries: WeekBoundaries,
    config: PatchTSTConfig,
) -> list[SymbolPrediction]:
    """Run PatchTST inference -- single forward pass, 5-day close prediction.

    Single forward pass produces (batch, 5, 1) output. RevIN automatically
    denormalizes output to original log-return scale. Extract close_ret
    (index 0) for weekly return. NO scaler inverse-transform needed.

    Args:
        model: Loaded PatchTSTForPrediction model in eval mode
        feature_scaler: Fitted StandardScaler (diagnostic only -- NOT used here)
        features_list: List of InferenceFeatures (one per symbol)
        week_boundaries: Target week info for the response
        config: PatchTST configuration (for feature names)

    Returns:
        List of SymbolPrediction results
    """
    predictions = []

    valid_features = [(f.symbol, f) for f in features_list if f.features is not None]
    invalid_features = [f for f in features_list if f.features is None]

    # Handle symbols without enough data
    for feat in invalid_features:
        predictions.append(
            SymbolPrediction(
                symbol=feat.symbol,
                predicted_weekly_return_pct=None,
                direction="FLAT",
                has_enough_history=False,
                history_days_used=feat.history_days_used,
                data_end_date=feat.data_end_date.isoformat()
                if feat.data_end_date
                else None,
                target_week_start=week_boundaries.target_week_start.isoformat(),
                target_week_end=week_boundaries.target_week_end.isoformat(),
                daily_returns=None,
            )
        )

    if not valid_features:
        return predictions

    close_ret_idx = config.feature_names.index("close_ret")

    # Prepare input batch: (n_samples, context_length, n_channels)
    X_batch = np.array([f.features for _, f in valid_features])

    model.eval()
    device = next(model.parameters()).device

    # Single forward pass -- NO scaler transform, RevIN normalizes internally
    # Output is (batch, 5, 5) already in ORIGINAL scale (denormalized by RevIN)
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_batch).float().to(device)
        outputs = model(past_values=X_tensor).prediction_outputs
        # Extract close_ret channel -- already in log-return scale (denormalized by RevIN)
        daily_preds = outputs[:, :, close_ret_idx].cpu().numpy()  # (batch, 5)
        del X_tensor, outputs

    # NO inverse-transform needed! prediction_outputs are denormalized by RevIN

    # Build prediction results
    for i, (symbol, feat) in enumerate(valid_features):
        symbol_daily = daily_preds[i]  # (5,) daily close log returns

        # Compound next-prediction_length trading-day close log returns
        # (API field name remains predicted_weekly_return_pct).
        weekly_return = float(np.exp(np.sum(symbol_daily)) - 1)

        # Daily returns list for response
        daily_returns_list = symbol_daily.tolist()

        weekly_return_pct = weekly_return * 100
        direction = classify_direction(weekly_return)

        predictions.append(
            SymbolPrediction(
                symbol=symbol,
                predicted_weekly_return_pct=round(weekly_return_pct, 4),
                direction=direction,
                has_enough_history=True,
                history_days_used=feat.history_days_used,
                data_end_date=feat.data_end_date.isoformat()
                if feat.data_end_date
                else None,
                target_week_start=week_boundaries.target_week_start.isoformat(),
                target_week_end=week_boundaries.target_week_end.isoformat(),
                daily_returns=daily_returns_list,
            )
        )

    return predictions


@dataclass
class BatchInferenceResult:
    """Result of batch PatchTST inference across multiple symbols."""

    predictions: list[SymbolPrediction]
    model_version: str


def run_batch_inference(
    symbols: list[str],
    cutoff_date: date,
    storage: PatchTSTModelStorage | None = None,
    artifacts: Any = None,
    exchange: str = "XNYS",
) -> BatchInferenceResult:
    """Run PatchTST inference on arbitrary symbols (close-only).

    End-to-end pipeline: load model -> fetch prices -> build features -> run model.
    Predictions are sorted by predicted_weekly_return_pct descending.

    ``predicted_weekly_return_pct`` is the compounded next-
    ``prediction_length`` trading-day close log-return x 100 (not necessarily
    the calendar Mon-Fri session count).

    Args:
        symbols: Ticker symbols to run inference on.
        cutoff_date: Friday cutoff date. Target week is the week AFTER this Friday.
        storage: Optional PatchTST storage. Used only when ``artifacts``
            is not supplied. Defaults to ``PatchTSTModelStorage()`` so
            legacy callers keep working.
        artifacts: Optional pre-loaded ``PatchTSTArtifacts``. When set,
            we skip the storage read entirely. Routes that go through
            the storage-policy helper pass artifacts directly so
            ``hf_first`` callers don't double-touch HuggingFace.
        exchange: exchange_calendars name for target-week boundaries
            (``XNYS`` US, ``XBOM`` India).

    Returns:
        BatchInferenceResult with sorted predictions and model version.

    Raises:
        ValueError: If no current PatchTST model is promoted (only
            when ``artifacts`` is not supplied).
    """
    from brain_api.core.prices import load_prices_yfinance
    from brain_api.storage.local import PatchTSTModelStorage as _DefaultStorage

    if artifacts is None:
        if storage is None:
            storage = _DefaultStorage()

        version = storage.read_current_version()
        if not version:
            raise ValueError("No current PatchTST model version available")

        artifacts = storage.load_current_artifacts()
    config = artifacts.config

    week_boundaries = compute_week_from_cutoff(cutoff_date, exchange=exchange)
    logger.info(
        f"[PatchTST batch] {len(symbols)} symbols, "
        f"cutoff={cutoff_date}, target={week_boundaries.target_week_start}..{week_boundaries.target_week_end}"
    )

    buffer_days = config.context_length * 2 + 30
    data_start = week_boundaries.target_week_start - timedelta(days=buffer_days)
    data_end = week_boundaries.target_week_start - timedelta(days=1)

    prices = load_prices_yfinance(symbols, data_start, data_end)

    features_list: list[InferenceFeatures] = []
    for symbol in symbols:
        prices_df = prices.get(symbol)
        if prices_df is None or prices_df.empty:
            features_list.append(
                InferenceFeatures(
                    symbol=symbol,
                    features=None,
                    has_enough_history=False,
                    history_days_used=0,
                    data_end_date=None,
                    starting_price=None,
                )
            )
        else:
            features_list.append(
                build_inference_features(
                    symbol=symbol,
                    prices_df=prices_df,
                    config=config,
                    cutoff_date=week_boundaries.target_week_start,
                )
            )

    predictions = run_inference(
        model=artifacts.model,
        feature_scaler=artifacts.feature_scaler,
        features_list=features_list,
        week_boundaries=week_boundaries,
        config=config,
    )

    valid = [p for p in predictions if p.predicted_weekly_return_pct is not None]
    invalid = [p for p in predictions if p.predicted_weekly_return_pct is None]
    sorted_predictions = (
        sorted(
            valid,
            key=lambda p: p.predicted_weekly_return_pct,  # type: ignore[arg-type]
            reverse=True,
        )
        + invalid
    )

    logger.info(f"[PatchTST batch] Done: {len(valid)} valid / {len(symbols)} total")

    return BatchInferenceResult(
        predictions=sorted_predictions,
        model_version=artifacts.version,
    )

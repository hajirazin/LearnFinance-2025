"""PatchTST inference helpers.

5-channel OHLCV direct 5-day multi-task inference.

Single forward pass produces (batch, 5, 5) output -- 5 days x 5 channels.
RevIN automatically denormalizes output to original log-return scale.
Extract close_ret channel for weekly return prediction. NO inverse-transform
needed.
"""

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from transformers import PatchTSTForPrediction

from brain_api.core.features import compute_ohlcv_log_returns
from brain_api.core.inference_utils import WeekBoundaries
from brain_api.core.model_types import classify_direction
from brain_api.core.patchtst.config import PatchTSTConfig


@dataclass
class InferenceFeatures:
    """Multi-channel features prepared for inference for a single symbol."""

    symbol: str
    features: np.ndarray | None  # Shape: (context_length, 5) -- OHLCV only, or None
    has_enough_history: bool
    history_days_used: int
    data_end_date: date | None
    has_news_data: bool
    has_fundamentals_data: bool
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
    has_news_data: bool
    has_fundamentals_data: bool
    daily_returns: list[float] | None = None  # 5 daily close_ret predictions


def build_inference_features(
    symbol: str,
    prices_df: pd.DataFrame,
    news_df: pd.DataFrame | None,
    fundamentals_df: pd.DataFrame | None,
    config: PatchTSTConfig,
    cutoff_date: date,
) -> InferenceFeatures:
    """Build 5-channel OHLCV feature sequence for inference.

    Still loads news/fundamentals data to set has_news_data/has_fundamentals_data
    flags in the response. But only OHLCV log returns (5 channels) are returned
    as model features.

    Args:
        symbol: Ticker symbol
        prices_df: DataFrame with OHLCV columns and DatetimeIndex
        news_df: DataFrame with 'sentiment_score' column and DatetimeIndex (or None)
        fundamentals_df: DataFrame with fundamental ratio columns and DatetimeIndex (or None)
        config: PatchTST config with context_length and feature settings
        cutoff_date: Features end before this date (typically target_week_start)

    Returns:
        InferenceFeatures with 5-channel OHLCV feature sequence (UNSCALED)
    """
    has_news_data = news_df is not None and len(news_df) > 0
    has_fundamentals_data = fundamentals_df is not None and len(fundamentals_df) > 0

    if prices_df.empty:
        return InferenceFeatures(
            symbol=symbol,
            features=None,
            has_enough_history=False,
            history_days_used=0,
            data_end_date=None,
            has_news_data=has_news_data,
            has_fundamentals_data=has_fundamentals_data,
            starting_price=None,
        )

    if not isinstance(prices_df.index, pd.DatetimeIndex):
        return InferenceFeatures(
            symbol=symbol,
            features=None,
            has_enough_history=False,
            history_days_used=0,
            data_end_date=None,
            has_news_data=has_news_data,
            has_fundamentals_data=has_fundamentals_data,
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
            has_news_data=has_news_data,
            has_fundamentals_data=has_fundamentals_data,
            starting_price=None,
        )

    # Compute OHLCV log returns (5 channels)
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
            has_news_data=has_news_data,
            has_fundamentals_data=has_fundamentals_data,
            starting_price=None,
        )

    # Extract only 5 OHLCV columns for model input
    ohlcv_cols = ["open_ret", "high_ret", "low_ret", "close_ret", "volume_ret"]
    sequence = features_df[ohlcv_cols].iloc[-config.context_length :].values  # (60, 5)
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
        features=sequence,  # (context_length, 5) -- OHLCV only, UNSCALED
        has_enough_history=True,
        history_days_used=len(features_df),
        data_end_date=data_end_date,
        has_news_data=has_news_data,
        has_fundamentals_data=has_fundamentals_data,
        starting_price=starting_price,
    )


def run_inference(
    model: PatchTSTForPrediction,
    feature_scaler: StandardScaler,
    features_list: list[InferenceFeatures],
    week_boundaries: WeekBoundaries,
    config: PatchTSTConfig,
) -> list[SymbolPrediction]:
    """Run PatchTST inference -- single forward pass, 5-day direct prediction.

    Single forward pass produces (batch, 5, 5) output. RevIN automatically
    denormalizes output to original log-return scale. Extract close_ret
    channel for weekly return. NO scaler inverse-transform needed.

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
                has_news_data=feat.has_news_data,
                has_fundamentals_data=feat.has_fundamentals_data,
                daily_returns=None,
            )
        )

    if not valid_features:
        return predictions

    # close_ret channel index in the 5-channel OHLCV output
    close_ret_idx = config.feature_names.index("close_ret")  # = 3

    # Prepare input batch: (n_samples, context_length, 5) -- raw OHLCV log returns
    X_batch = np.array([f.features for _, f in valid_features])

    model.eval()
    device = next(model.parameters()).device

    # Single forward pass -- NO scaler transform, RevIN normalizes internally
    # Output is (batch, 5, 5) already in ORIGINAL scale (denormalized by RevIN)
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_batch).float().to(device)  # (batch, 60, 5)
        outputs = model(
            past_values=X_tensor
        ).prediction_outputs  # (batch, 5, 5) original scale
        # Extract close_ret channel -- already in log-return scale (denormalized by RevIN)
        daily_preds = outputs[:, :, close_ret_idx].cpu().numpy()  # (batch, 5)
        del X_tensor, outputs

    # NO inverse-transform needed! prediction_outputs are denormalized by RevIN

    # Build prediction results
    for i, (symbol, feat) in enumerate(valid_features):
        symbol_daily = daily_preds[i]  # (5,) daily close log returns

        # Compound log returns: weekly_return = exp(sum(5 log returns)) - 1
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
                has_news_data=feat.has_news_data,
                has_fundamentals_data=feat.has_fundamentals_data,
                daily_returns=daily_returns_list,
            )
        )

    return predictions

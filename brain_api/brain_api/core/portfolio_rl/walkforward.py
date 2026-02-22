"""Walk-forward forecast generation for RL training.

Generates forecast features without look-ahead bias by using
forecaster models trained only on prior data.
"""

import logging
import threading
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SnapshotUnavailableError(Exception):
    """Raised when a required snapshot is not available locally or on HuggingFace."""


class SnapshotInferenceError(Exception):
    """Raised when snapshot inference fails for a symbol/year."""


def generate_walkforward_forecasts(
    weekly_prices: dict[str, np.ndarray],
    weekly_dates: pd.DatetimeIndex,
    symbols: list[str],
    forecaster_type: Literal["lstm", "patchtst"],
    shutdown_event: threading.Event | None = None,
) -> dict[str, np.ndarray]:
    """Generate walk-forward forecasts using trained LSTM/PatchTST snapshots.

    For each year, loads a forecaster trained on prior data (snapshot for
    Dec 31 of the previous year) and generates predictions.

    Raises SnapshotUnavailableError if any required snapshot is missing.
    Raises SnapshotInferenceError if inference fails for any symbol/year.

    Args:
        weekly_prices: Dict of symbol -> weekly price array
        weekly_dates: DatetimeIndex corresponding to weekly_prices
        symbols: Ordered list of symbols
        forecaster_type: "lstm" or "patchtst"
        shutdown_event: If set, the function returns partial results.

    Raises:
        SnapshotUnavailableError: If any required snapshot is missing.
        SnapshotInferenceError: If inference fails.

    Returns:
        Dict of symbol -> array of forecast values
    """
    from brain_api.storage.forecaster_snapshots import SnapshotLocalStorage

    snapshot_storage = SnapshotLocalStorage(forecaster_type)

    forecasts: dict[str, np.ndarray] = {}

    if len(weekly_dates) == 0:
        return forecasts

    start_year = weekly_dates[0].year
    end_year = weekly_dates[-1].year

    # Group weeks by year
    year_groups = {}
    for i, dt in enumerate(weekly_dates):
        year = dt.year
        if year not in year_groups:
            year_groups[year] = []
        year_groups[year].append(i)

    # Pre-check: ensure ALL snapshots are available.
    # Fail fast if any snapshot is missing.
    for year in range(start_year, end_year + 1):
        if year not in year_groups:
            continue
        cutoff_date = date(year - 1, 12, 31)
        if not snapshot_storage.ensure_snapshot_available(cutoff_date):
            raise SnapshotUnavailableError(
                f"Snapshot for {cutoff_date} ({forecaster_type}) not available "
                f"locally or on HuggingFace. Run training backfill first."
            )

    for sym_idx, symbol in enumerate(symbols):
        if shutdown_event and shutdown_event.is_set():
            logger.warning(
                f"[WalkForward] Shutdown requested after {sym_idx}/{len(symbols)} symbols"
            )
            break

        if symbol not in weekly_prices:
            continue

        prices = weekly_prices[symbol]
        n_weeks = len(prices)

        if n_weeks < 2:
            continue

        symbol_forecasts = np.zeros(n_weeks - 1)

        for year in range(start_year, end_year + 1):
            if year not in year_groups:
                continue

            year_indices = year_groups[year]
            cutoff_date = date(year - 1, 12, 31)
            snapshot_path = snapshot_storage._snapshot_path(cutoff_date)
            preds = _run_snapshot_inference(
                snapshot_path,
                forecaster_type,
                symbol,
                year_indices,
                weekly_dates,
            )
            for idx, pred in zip(year_indices, preds, strict=False):
                if idx < n_weeks - 1:
                    symbol_forecasts[idx] = pred

        forecasts[symbol] = symbol_forecasts

    return forecasts


def _run_snapshot_inference(
    snapshot_path: Path,
    forecaster_type: str,
    symbol: str,
    year_indices: list[int],
    weekly_dates: pd.DatetimeIndex,
) -> list[float]:
    """Run inference using a snapshot model.

    Loads a pre-trained LSTM or PatchTST snapshot and generates
    predictions for each week in the target year.

    Args:
        snapshot_path: Path to snapshot directory
        forecaster_type: "lstm" or "patchtst"
        symbol: Stock symbol
        year_indices: Indices for the target year
        weekly_dates: DatetimeIndex of weekly dates

    Returns:
        List of predictions for each week in year_indices
    """
    from brain_api.storage.forecaster_snapshots import SnapshotLocalStorage

    # Load snapshot artifacts
    cutoff_date_str = snapshot_path.name.replace("snapshot-", "")
    cutoff_date = date.fromisoformat(cutoff_date_str)

    storage = SnapshotLocalStorage(forecaster_type)
    artifacts = storage.load_snapshot(cutoff_date)

    if forecaster_type == "lstm":
        predictions = _run_lstm_snapshot_inference(
            artifacts,
            year_indices,
            weekly_dates=weekly_dates,
            symbol=symbol,
        )
    else:
        predictions = _run_patchtst_snapshot_inference(
            artifacts,
            year_indices,
            weekly_dates=weekly_dates,
            symbol=symbol,
        )

    return predictions


def _run_lstm_snapshot_inference(
    artifacts: "LSTMSnapshotArtifacts",
    year_indices: list[int],
    weekly_dates: pd.DatetimeIndex | None = None,
    symbol: str | None = None,
) -> list[float]:
    """Run LSTM snapshot inference for a symbol using direct 5-day prediction.

    Uses single forward pass for 5 close log returns (no autoregressive loop).

    Args:
        artifacts: Loaded LSTM snapshot artifacts
        year_indices: Indices to predict
        weekly_dates: DatetimeIndex of weekly dates (for loading daily data)
        symbol: Stock symbol (for loading daily OHLCV)

    Returns:
        List of predictions
    """
    import torch

    predictions = []
    model = artifacts.model
    scaler = artifacts.feature_scaler
    config = artifacts.config
    seq_len = config.sequence_length

    if weekly_dates is None or symbol is None:
        raise SnapshotInferenceError(
            f"LSTM inference requires weekly_dates and symbol, got "
            f"weekly_dates={weekly_dates is not None}, symbol={symbol}"
        )

    if len(year_indices) == 0:
        return predictions

    min_idx = min(year_indices)
    max_idx = max(year_indices)
    buffer_days = seq_len * 2 + 30

    start_date = weekly_dates[max(0, min_idx - 10)].date() - timedelta(days=buffer_days)
    end_date = weekly_dates[min(max_idx, len(weekly_dates) - 1)].date()

    daily_ohlcv = _load_daily_ohlcv(symbol, start_date, end_date)
    if daily_ohlcv is None:
        raise SnapshotInferenceError(
            f"Failed to load daily OHLCV for {symbol} ({start_date} to {end_date})"
        )

    logger.debug(f"[WalkForward] Loaded {len(daily_ohlcv)} days of OHLCV for {symbol}")

    model.eval()

    with torch.no_grad():
        for i in year_indices:
            ret = _predict_single_week_lstm(
                model=model,
                scaler=scaler,
                config=config,
                weekly_idx=i,
                weekly_dates=weekly_dates,
                daily_ohlcv=daily_ohlcv,
            )
            predictions.append(ret)

    return predictions


def _predict_single_week_lstm(
    model,
    scaler,
    config,
    weekly_idx: int,
    weekly_dates: pd.DatetimeIndex,
    daily_ohlcv: pd.DataFrame,
) -> float:
    """Generate LSTM weekly prediction using direct 5-day forward pass.

    Single forward pass predicts 5 daily close log returns. Weekly return
    is computed by compounding: exp(sum(5 log returns)) - 1.

    NO inverse transform is applied -- targets are never scaled during
    training, so the model outputs raw log returns directly (Bug #1 fix).

    Input includes anchor day's return (last row of features_df), matching
    the corrected training alignment (Bug #2 fix).

    Raises:
        SnapshotInferenceError: If insufficient data for prediction.

    Returns:
        Weekly return prediction.
    """
    import torch

    from brain_api.core.features import compute_ohlcv_log_returns

    seq_len = config.sequence_length

    week_date = weekly_dates[weekly_idx]
    cutoff = week_date.date()

    if daily_ohlcv.index.tz is not None:
        cutoff_ts = pd.Timestamp(cutoff).tz_localize(daily_ohlcv.index.tz)
    else:
        cutoff_ts = pd.Timestamp(cutoff)

    ohlcv_subset = daily_ohlcv[daily_ohlcv.index < cutoff_ts].copy()

    if len(ohlcv_subset) < seq_len + 1:
        raise SnapshotInferenceError(
            f"Insufficient daily data for LSTM: need {seq_len + 1}, "
            f"got {len(ohlcv_subset)} (cutoff={cutoff})"
        )

    features_df = compute_ohlcv_log_returns(
        ohlcv_subset, use_returns=config.use_returns
    )

    if len(features_df) < seq_len:
        raise SnapshotInferenceError(
            f"Insufficient features after log-return computation: "
            f"need {seq_len}, got {len(features_df)} (cutoff={cutoff})"
        )

    features_df = features_df.iloc[-seq_len:]
    features = features_df.values

    if features.shape[1] != 5:
        raise SnapshotInferenceError(
            f"Expected 5 OHLCV features, got {features.shape[1]} (cutoff={cutoff})"
        )

    features_scaled = scaler.transform(features) if scaler is not None else features

    x = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(0)
    output = model(x)  # (1, 5)
    daily_log_returns = output.squeeze().cpu().numpy()  # (5,)

    # NO inverse transform -- targets are never scaled during training.
    weekly_return = float(np.exp(np.sum(daily_log_returns)) - 1)
    return weekly_return


def _run_patchtst_snapshot_inference(
    artifacts: "PatchTSTSnapshotArtifacts",
    year_indices: list[int],
    weekly_dates: pd.DatetimeIndex | None = None,
    symbol: str | None = None,
) -> list[float]:
    """Run PatchTST snapshot inference for a symbol.

    Loads daily OHLCV data only (no news/fundamentals -- PatchTST uses
    5-channel OHLCV input). Single forward pass per week produces
    direct 5-day predictions.

    Args:
        artifacts: Loaded PatchTST snapshot artifacts
        year_indices: Indices to predict
        weekly_dates: DatetimeIndex of weekly dates (for loading daily data)
        symbol: Stock symbol (for loading historical OHLCV)

    Raises:
        SnapshotInferenceError: If daily OHLCV data cannot be loaded.

    Returns:
        List of predictions
    """
    import torch

    predictions = []
    model = artifacts.model
    scaler = artifacts.feature_scaler
    config = artifacts.config
    context_length = config.context_length

    if weekly_dates is None or symbol is None:
        raise SnapshotInferenceError(
            f"PatchTST inference requires weekly_dates and symbol, got "
            f"weekly_dates={weekly_dates is not None}, symbol={symbol}"
        )

    if len(year_indices) == 0:
        return predictions

    min_idx = min(year_indices)
    max_idx = max(year_indices)
    buffer_days = context_length * 2 + 30

    start_date = weekly_dates[max(0, min_idx - 10)].date() - timedelta(days=buffer_days)
    end_date = weekly_dates[min(max_idx, len(weekly_dates) - 1)].date()

    daily_ohlcv = _load_daily_ohlcv(symbol, start_date, end_date)
    if daily_ohlcv is None:
        raise SnapshotInferenceError(
            f"Failed to load daily OHLCV for {symbol} ({start_date} to {end_date})"
        )

    logger.debug(f"[WalkForward] Loaded {len(daily_ohlcv)} days of OHLCV for {symbol}")

    model.eval()

    with torch.no_grad():
        for i in year_indices:
            ret = _predict_single_week_patchtst(
                model=model,
                scaler=scaler,
                config=config,
                weekly_idx=i,
                weekly_dates=weekly_dates,
                daily_ohlcv=daily_ohlcv,
            )
            predictions.append(ret)

    return predictions


def _predict_single_week_patchtst(
    model,
    scaler,
    config,
    weekly_idx: int,
    weekly_dates: pd.DatetimeIndex,
    daily_ohlcv: pd.DataFrame,
) -> float:
    """Generate PatchTST weekly prediction using direct 5-day forecasting.

    Single forward pass produces (1, 5, 5) output -- 5 days x 5 channels.
    RevIN denormalizes output to original log-return scale automatically.
    Extract close_ret channel. NO scaler inverse-transform needed.

    Bug fixes applied:
    - Bug #E: Removed .permute(0, 2, 1) that crashed PatchTST
    - Bug #F: Uses exp(sum(log_returns)) - 1 instead of prod(1 + log_returns) - 1

    Raises:
        SnapshotInferenceError: If insufficient data for prediction.

    Returns:
        Weekly return prediction.
    """
    import torch

    from brain_api.core.features import compute_ohlcv_log_returns

    context_length = config.context_length

    week_date = weekly_dates[weekly_idx]
    cutoff = week_date.date()

    if daily_ohlcv.index.tz is not None:
        cutoff_ts = pd.Timestamp(cutoff).tz_localize(daily_ohlcv.index.tz)
    else:
        cutoff_ts = pd.Timestamp(cutoff)

    ohlcv_subset = daily_ohlcv[daily_ohlcv.index < cutoff_ts].copy()

    if len(ohlcv_subset) < context_length:
        raise SnapshotInferenceError(
            f"Insufficient daily data for PatchTST: need {context_length}, "
            f"got {len(ohlcv_subset)} (cutoff={cutoff})"
        )

    features_df = compute_ohlcv_log_returns(
        ohlcv_subset, use_returns=config.use_returns
    )
    if features_df.index.tz is not None:
        features_df.index = features_df.index.tz_localize(None)

    if len(features_df) < context_length:
        raise SnapshotInferenceError(
            f"Insufficient features after log-return computation for PatchTST: "
            f"need {context_length}, got {len(features_df)} (cutoff={cutoff})"
        )

    ohlcv_cols = ["open_ret", "high_ret", "low_ret", "close_ret", "volume_ret"]
    ohlcv_features = (
        features_df[ohlcv_cols].iloc[-context_length:].values
    )  # (context_length, 5)

    close_ret_idx = 3

    # NO scaler transform -- RevIN normalizes internally
    # Single forward pass -- NO permute! (Bug #E fix)
    x = torch.tensor(ohlcv_features, dtype=torch.float32).unsqueeze(
        0
    )  # (1, context_length, 5)
    output = model(past_values=x).prediction_outputs  # (1, 5, 5) ORIGINAL scale

    daily_log_returns = output[0, :, close_ret_idx].cpu().numpy()  # (5,)

    # NO inverse-transform needed! RevIN already denormalized
    # Bug #F fix: exp(sum) not prod(1+)
    weekly_return = float(np.exp(np.sum(daily_log_returns)) - 1)
    return weekly_return


def _load_daily_ohlcv(
    symbol: str, start_date: date, end_date: date
) -> pd.DataFrame | None:
    """Load daily OHLCV data for a symbol."""
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start_date.isoformat(),
            end=(end_date + timedelta(days=1)).isoformat(),
            interval="1d",
        )
        if df.empty:
            return None

        # Normalize column names
        df.columns = df.columns.str.lower()
        return df[["open", "high", "low", "close", "volume"]]
    except Exception as e:
        logger.debug(f"[WalkForward] Failed to load OHLCV for {symbol}: {e}")
        return None


# Type hints for snapshot artifacts
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brain_api.storage.forecaster_snapshots import (
        LSTMSnapshotArtifacts,
        PatchTSTSnapshotArtifacts,
    )


def build_forecast_features(
    weekly_prices: dict[str, np.ndarray],
    weekly_dates: pd.DatetimeIndex,
    symbols: list[str],
    forecaster_type: Literal["lstm", "patchtst"] = "lstm",
    shutdown_event: threading.Event | None = None,
) -> dict[str, np.ndarray]:
    """Build forecast features for RL training.

    Requires pre-trained model snapshots. Raises SnapshotUnavailableError
    if any required snapshot is missing.

    Args:
        weekly_prices: Dict of symbol -> weekly price array
        weekly_dates: DatetimeIndex of weekly dates
        symbols: List of symbols
        forecaster_type: Which forecaster to use
        shutdown_event: If set, the function returns partial results.

    Raises:
        SnapshotUnavailableError: If any required snapshot is missing.
        SnapshotInferenceError: If inference fails for any symbol/year.

    Returns:
        Dict of symbol -> array of forecast values
    """
    print(f"[PortfolioRL] Generating walk-forward forecasts ({forecaster_type})...")

    forecasts = generate_walkforward_forecasts(
        weekly_prices,
        weekly_dates,
        symbols,
        forecaster_type,
        shutdown_event=shutdown_event,
    )

    print(f"[PortfolioRL] Generated forecasts for {len(forecasts)} symbols")
    return forecasts


def build_dual_forecast_features(
    weekly_prices: dict[str, np.ndarray],
    weekly_dates: pd.DatetimeIndex,
    symbols: list[str],
    shutdown_event: threading.Event | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Build both LSTM and PatchTST forecast features for RL training.

    Requires pre-trained snapshots for both LSTM and PatchTST.

    Args:
        weekly_prices: Dict of symbol -> weekly price array
        weekly_dates: DatetimeIndex of weekly dates
        symbols: List of symbols
        shutdown_event: If set, returns partial results.

    Raises:
        SnapshotUnavailableError: If any required snapshot is missing.
        SnapshotInferenceError: If inference fails for any symbol/year.

    Returns:
        Tuple of (lstm_forecasts, patchtst_forecasts) where each is Dict of symbol -> array
    """
    print("[PortfolioRL] Generating dual walk-forward forecasts (LSTM + PatchTST)...")

    lstm_forecasts = build_forecast_features(
        weekly_prices,
        weekly_dates,
        symbols,
        forecaster_type="lstm",
        shutdown_event=shutdown_event,
    )

    patchtst_forecasts = build_forecast_features(
        weekly_prices,
        weekly_dates,
        symbols,
        forecaster_type="patchtst",
        shutdown_event=shutdown_event,
    )

    print(f"[PortfolioRL] Generated dual forecasts for {len(lstm_forecasts)} symbols")
    return (lstm_forecasts, patchtst_forecasts)

"""Walk-forward forecast generation for RL training.

Generates forecast features without look-ahead bias by using
forecaster models trained only on prior data.
"""

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_momentum_proxy(
    prices: np.ndarray,
    lookback_weeks: int = 4,
) -> np.ndarray:
    """Compute simple momentum as a proxy for forecast during bootstrap period.

    Args:
        prices: Weekly price array
        lookback_weeks: Number of weeks for momentum calculation

    Returns:
        Array of momentum values (same length as prices)
    """
    if len(prices) < lookback_weeks + 1:
        return np.zeros(len(prices))

    # Simple momentum: return over lookback period
    momentum = np.zeros(len(prices))
    for i in range(lookback_weeks, len(prices)):
        if prices[i - lookback_weeks] > 0:
            momentum[i] = (prices[i] - prices[i - lookback_weeks]) / prices[
                i - lookback_weeks
            ]

    return momentum


def get_year_mask(weekly_dates: pd.DatetimeIndex, year: int) -> np.ndarray:
    """Get boolean mask for weeks in a specific year.

    Args:
        weekly_dates: DatetimeIndex of weekly dates
        year: Target year

    Returns:
        Boolean mask array
    """
    return (weekly_dates.year == year).values


def generate_walkforward_forecasts_simple(
    weekly_prices: dict[str, np.ndarray],
    weekly_dates: pd.DatetimeIndex,
    symbols: list[str],
    bootstrap_years: int = 4,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Generate walk-forward forecasts using simple momentum proxy.

    For v1, we use momentum as a proxy for forecast features during training.
    This avoids the complexity of training/caching LSTM snapshots while still
    providing a meaningful "alpha" signal without look-ahead bias.

    The momentum at week t is computed from prices up to week t-1 only.

    Args:
        weekly_prices: Dict of symbol -> weekly price array
        weekly_dates: DatetimeIndex corresponding to weekly_prices
        symbols: Ordered list of symbols
        bootstrap_years: First N years use zeros (truly no forecast)

    Returns:
        Tuple of (forecasts, volatilities) where each is
        Dict of symbol -> array of values (same length as prices - 1).
        Volatilities are all zeros since momentum has no daily return path.
    """
    forecasts: dict[str, np.ndarray] = {}
    volatilities: dict[str, np.ndarray] = {}

    if len(weekly_dates) == 0:
        return (forecasts, volatilities)

    start_year = weekly_dates[0].year
    weekly_dates[-1].year

    for symbol in symbols:
        if symbol not in weekly_prices:
            continue

        prices = weekly_prices[symbol]
        n_weeks = len(prices)

        if n_weeks < 2:
            continue

        # Initialize forecasts (length = n_weeks - 1, for returns)
        symbol_forecasts = np.zeros(n_weeks - 1)
        # Momentum proxy has no daily returns, so volatility is always 0
        symbol_volatilities = np.zeros(n_weeks - 1)

        for i in range(n_weeks - 1):
            week_date = weekly_dates[i]
            year = week_date.year

            if year < start_year + bootstrap_years:
                # Bootstrap period: use zeros
                symbol_forecasts[i] = 0.0
            else:
                # Post-bootstrap: use momentum as proxy
                # Momentum computed from prices up to and including this week
                lookback = 4  # 4-week momentum
                if i >= lookback:
                    if prices[i - lookback] > 0:
                        symbol_forecasts[i] = (
                            prices[i] - prices[i - lookback]
                        ) / prices[i - lookback]

        forecasts[symbol] = symbol_forecasts
        volatilities[symbol] = symbol_volatilities

    return (forecasts, volatilities)


def generate_walkforward_forecasts_with_model(
    weekly_prices: dict[str, np.ndarray],
    weekly_dates: pd.DatetimeIndex,
    symbols: list[str],
    forecaster_type: Literal["lstm", "patchtst"],
    bootstrap_years: int = 4,
    snapshot_dir: Path | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Generate walk-forward forecasts using trained LSTM/PatchTST snapshots.

    For each year after bootstrap, loads a forecaster trained on prior data
    and generates predictions. Requires pre-computed snapshots.

    Snapshots are loaded from local storage. If HuggingFace is configured and
    a snapshot is not available locally, it will be downloaded from HF.

    Args:
        weekly_prices: Dict of symbol -> weekly price array
        weekly_dates: DatetimeIndex corresponding to weekly_prices
        symbols: Ordered list of symbols
        forecaster_type: "lstm" or "patchtst"
        bootstrap_years: First N years use momentum proxy
        snapshot_dir: Directory containing forecaster snapshots (for testing)

    Returns:
        Tuple of (forecasts, volatilities) where each is Dict of symbol -> array
    """
    from brain_api.storage.forecaster_snapshots import SnapshotLocalStorage

    # Create snapshot storage for this forecaster type
    snapshot_storage = SnapshotLocalStorage(forecaster_type)

    # Check if any snapshots exist locally (we'll try HF later if needed)
    if snapshot_dir is None:
        from brain_api.core.portfolio_rl.data_loading import get_default_data_path

        # Snapshots are stored alongside main model: models/{type}/snapshots/
        snapshot_dir = (
            get_default_data_path() / "models" / forecaster_type / "snapshots"
        )

    forecasts: dict[str, np.ndarray] = {}
    volatilities: dict[str, np.ndarray] = {}

    if len(weekly_dates) == 0:
        return (forecasts, volatilities)

    start_year = weekly_dates[0].year
    end_year = weekly_dates[-1].year

    # Group weeks by year
    year_groups = {}
    for i, dt in enumerate(weekly_dates):
        year = dt.year
        if year not in year_groups:
            year_groups[year] = []
        year_groups[year].append(i)

    for symbol in symbols:
        if symbol not in weekly_prices:
            continue

        prices = weekly_prices[symbol]
        n_weeks = len(prices)

        if n_weeks < 2:
            continue

        symbol_forecasts = np.zeros(n_weeks - 1)
        symbol_volatilities = np.zeros(n_weeks - 1)

        for year in range(start_year, end_year + 1):
            if year not in year_groups:
                continue

            year_indices = year_groups[year]

            if year < start_year + bootstrap_years:
                # Bootstrap: use momentum (volatility = 0 since no daily path)
                for i in year_indices:
                    if i < n_weeks - 1:
                        lookback = 4
                        if i >= lookback and prices[i - lookback] > 0:
                            symbol_forecasts[i] = (
                                prices[i] - prices[i - lookback]
                            ) / prices[i - lookback]
                        # volatility stays 0.0 for momentum
            else:
                # Try to load snapshot for this cutoff
                cutoff_date = date(year - 1, 12, 31)

                # Try to ensure snapshot is available (downloads from HF if needed)
                snapshot_available = snapshot_storage.ensure_snapshot_available(
                    cutoff_date
                )

                if snapshot_available:
                    try:
                        # Load and run inference
                        snapshot_path = (
                            snapshot_dir / f"snapshot_{cutoff_date.isoformat()}"
                        )
                        preds, vols = _run_snapshot_inference(
                            snapshot_path,
                            forecaster_type,
                            symbol,
                            prices,
                            year_indices,
                            weekly_dates=weekly_dates,
                        )
                        for idx, pred, vol in zip(
                            year_indices, preds, vols, strict=False
                        ):
                            if idx < n_weeks - 1:
                                symbol_forecasts[idx] = pred
                                symbol_volatilities[idx] = vol
                    except Exception as e:
                        print(
                            f"[WalkForward] Error running snapshot for {symbol} year {year}: {e}"
                        )
                        # Fallback to momentum (volatility = 0)
                        for i in year_indices:
                            if i < n_weeks - 1:
                                lookback = 4
                                if i >= lookback and prices[i - lookback] > 0:
                                    symbol_forecasts[i] = (
                                        prices[i] - prices[i - lookback]
                                    ) / prices[i - lookback]
                else:
                    # No snapshot available (locally or HF), use momentum
                    for i in year_indices:
                        if i < n_weeks - 1:
                            lookback = 4
                            if i >= lookback and prices[i - lookback] > 0:
                                symbol_forecasts[i] = (
                                    prices[i] - prices[i - lookback]
                                ) / prices[i - lookback]

        forecasts[symbol] = symbol_forecasts
        volatilities[symbol] = symbol_volatilities

    return (forecasts, volatilities)


def _run_snapshot_inference(
    snapshot_path: Path,
    forecaster_type: str,
    symbol: str,
    prices: np.ndarray,
    year_indices: list[int],
    weekly_dates: pd.DatetimeIndex | None = None,
) -> tuple[list[float], list[float]]:
    """Run inference using a snapshot model.

    Loads a pre-trained LSTM or PatchTST snapshot and generates
    predictions for each week in the target year.

    Args:
        snapshot_path: Path to snapshot directory
        forecaster_type: "lstm" or "patchtst"
        symbol: Stock symbol
        prices: Weekly prices array
        year_indices: Indices for the target year
        weekly_dates: DatetimeIndex of weekly dates (for multi-channel PatchTST)

    Returns:
        Tuple of (predictions, volatilities) lists for each week in year_indices
    """
    from brain_api.storage.forecaster_snapshots import SnapshotLocalStorage

    # Load snapshot artifacts
    cutoff_date_str = snapshot_path.name.replace("snapshot_", "")
    cutoff_date = date.fromisoformat(cutoff_date_str)

    storage = SnapshotLocalStorage(forecaster_type)
    artifacts = storage.load_snapshot(cutoff_date)

    if forecaster_type == "lstm":
        # Run LSTM inference (with multi-channel OHLCV support)
        predictions, volatilities = _run_lstm_snapshot_inference(
            artifacts,
            prices,
            year_indices,
            weekly_dates=weekly_dates,
            symbol=symbol,
        )
    else:
        # Run PatchTST inference (multi-channel when available)
        predictions, volatilities = _run_patchtst_snapshot_inference(
            artifacts,
            prices,
            year_indices,
            weekly_dates=weekly_dates,
            symbol=symbol,
        )

    return (predictions, volatilities)


def _run_lstm_snapshot_inference(
    artifacts: "LSTMSnapshotArtifacts",
    prices: np.ndarray,
    year_indices: list[int],
    weekly_dates: pd.DatetimeIndex | None = None,
    symbol: str | None = None,
) -> tuple[list[float], list[float]]:
    """Run LSTM snapshot inference for a symbol using direct 5-day prediction.

    Uses single forward pass for 5 close log returns (no autoregressive loop).
    Falls back to momentum if daily data unavailable.

    Args:
        artifacts: Loaded LSTM snapshot artifacts
        prices: Weekly price array for the symbol (fallback only)
        year_indices: Indices to predict
        weekly_dates: DatetimeIndex of weekly dates (for loading daily data)
        symbol: Stock symbol (for loading daily OHLCV)

    Returns:
        Tuple of (predictions, volatilities) lists
    """
    import torch

    predictions = []
    volatilities = []
    model = artifacts.model
    scaler = artifacts.feature_scaler
    config = artifacts.config
    seq_len = config.sequence_length

    # Momentum fallback helper - returns (return, 0.0 volatility)
    def momentum_fallback(idx: int) -> tuple[float, float]:
        lookback = 4
        if idx >= lookback and prices[idx - lookback] > 0:
            ret = (prices[idx] - prices[idx - lookback]) / prices[idx - lookback]
            return (ret, 0.0)
        return (0.0, 0.0)

    # Determine if we can use multi-channel features
    use_multichannel = weekly_dates is not None and symbol is not None

    # Load daily OHLCV if multi-channel is enabled
    daily_ohlcv: pd.DataFrame | None = None

    if use_multichannel and len(year_indices) > 0:
        try:
            # Get date range for this year (with buffer for context)
            min_idx = min(year_indices)
            max_idx = max(year_indices)
            buffer_days = seq_len * 2 + 30

            start_date = weekly_dates[max(0, min_idx - 10)].date() - timedelta(
                days=buffer_days
            )
            end_date = weekly_dates[min(max_idx, len(weekly_dates) - 1)].date()

            # Load daily OHLCV
            daily_ohlcv = _load_daily_ohlcv(symbol, start_date, end_date)

            if daily_ohlcv is not None:
                logger.debug(
                    f"[WalkForward] Loaded {len(daily_ohlcv)} days of OHLCV for {symbol}"
                )
        except Exception as e:
            logger.warning(
                f"[WalkForward] Failed to load daily OHLCV for {symbol}: {e}"
            )
            use_multichannel = False

    # If we couldn't load daily data, fall back to momentum for all predictions
    if not use_multichannel or daily_ohlcv is None:
        logger.debug(
            f"[WalkForward] Using momentum fallback for LSTM {symbol} "
            f"(multichannel={use_multichannel}, has_ohlcv={daily_ohlcv is not None})"
        )
        for i in year_indices:
            ret, vol = momentum_fallback(i)
            predictions.append(ret)
            volatilities.append(vol)
        return (predictions, volatilities)

    model.eval()

    with torch.no_grad():
        for i in year_indices:
            ret, vol = _predict_single_week_lstm(
                model=model,
                scaler=scaler,
                config=config,
                weekly_idx=i,
                weekly_prices=prices,
                weekly_dates=weekly_dates,
                daily_ohlcv=daily_ohlcv,
            )
            predictions.append(ret)
            volatilities.append(vol)

    return (predictions, volatilities)


def _predict_single_week_lstm(
    model,
    scaler,
    config,
    weekly_idx: int,
    weekly_prices: np.ndarray,
    weekly_dates: pd.DatetimeIndex,
    daily_ohlcv: pd.DataFrame,
) -> tuple[float, float]:
    """Generate LSTM weekly prediction using direct 5-day forward pass.

    Single forward pass predicts 5 daily close log returns. Weekly return
    is computed by compounding: exp(sum(5 log returns)) - 1.

    NO inverse transform is applied -- targets are never scaled during
    training, so the model outputs raw log returns directly (Bug #1 fix).

    Input includes anchor day's return (last row of features_df), matching
    the corrected training alignment (Bug #2 fix).

    Returns:
        Tuple of (weekly_return, volatility) where volatility is std of daily returns.
    """
    import torch

    from brain_api.core.features import compute_ohlcv_log_returns

    seq_len = config.sequence_length

    # Momentum fallback helper - returns (return, 0.0 volatility) since no daily path
    def momentum_fallback() -> tuple[float, float]:
        lookback = 4
        if weekly_idx >= lookback and weekly_prices[weekly_idx - lookback] > 0:
            ret = (
                weekly_prices[weekly_idx] - weekly_prices[weekly_idx - lookback]
            ) / weekly_prices[weekly_idx - lookback]
            return (ret, 0.0)
        return (0.0, 0.0)

    try:
        # Get cutoff date (Friday of this week - predict for next week)
        week_date = weekly_dates[weekly_idx]
        cutoff = week_date.date()

        # Filter daily data up to cutoff
        if daily_ohlcv.index.tz is not None:
            cutoff_ts = pd.Timestamp(cutoff).tz_localize(daily_ohlcv.index.tz)
        else:
            cutoff_ts = pd.Timestamp(cutoff)

        ohlcv_subset = daily_ohlcv[daily_ohlcv.index < cutoff_ts].copy()

        if len(ohlcv_subset) < seq_len + 1:  # +1 for returns computation
            return momentum_fallback()

        # Compute OHLCV log returns (5 features)
        features_df = compute_ohlcv_log_returns(
            ohlcv_subset, use_returns=config.use_returns
        )

        if len(features_df) < seq_len:
            return momentum_fallback()

        # Take last seq_len rows -- includes anchor day's return (last row),
        # matching the fixed training alignment (Bug #2 fix).
        features_df = features_df.iloc[-seq_len:]

        # Build feature array: (seq_len, 5)
        features = features_df.values

        # Verify we have 5 features
        if features.shape[1] != 5:
            logger.warning(
                f"[WalkForward] Expected 5 OHLCV features, got {features.shape[1]}"
            )
            return momentum_fallback()

        # Apply scaler
        features_scaled = scaler.transform(features) if scaler is not None else features

        # Single forward pass: model outputs (1, 5) = 5 close log returns
        x = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(0)
        output = model(x)  # (1, 5)
        daily_log_returns = output.squeeze().cpu().numpy()  # (5,)

        # NO inverse transform -- Bug #1 fix.
        # Targets are never scaled during training. Model outputs raw log returns.

        # Compound log returns to get weekly return: exp(sum) - 1
        weekly_return = float(np.exp(np.sum(daily_log_returns)) - 1)
        # Volatility as std of daily log return predictions
        volatility = float(np.std(daily_log_returns))
        return (weekly_return, volatility)

    except Exception as e:
        logger.debug(f"[WalkForward] LSTM direct 5-day inference failed: {e}")
        return momentum_fallback()


def _run_patchtst_snapshot_inference(
    artifacts: "PatchTSTSnapshotArtifacts",
    prices: np.ndarray,
    year_indices: list[int],
    weekly_dates: pd.DatetimeIndex | None = None,
    symbol: str | None = None,
) -> tuple[list[float], list[float]]:
    """Run PatchTST snapshot inference for a symbol.

    Loads daily OHLCV data only (no news/fundamentals -- PatchTST uses
    5-channel OHLCV input). Single forward pass per week produces
    direct 5-day predictions.

    Args:
        artifacts: Loaded PatchTST snapshot artifacts
        prices: Weekly price array for the symbol (fallback only)
        year_indices: Indices to predict
        weekly_dates: DatetimeIndex of weekly dates (for loading daily data)
        symbol: Stock symbol (for loading historical OHLCV)

    Returns:
        Tuple of (predictions, volatilities) lists
    """
    import torch

    predictions = []
    volatilities = []
    model = artifacts.model
    scaler = artifacts.feature_scaler
    config = artifacts.config
    context_length = config.context_length

    # Determine if we can use OHLCV features
    use_multichannel = weekly_dates is not None and symbol is not None

    # Load daily OHLCV only (skip news/fundamentals -- PatchTST uses 5 OHLCV channels)
    daily_ohlcv: pd.DataFrame | None = None

    if use_multichannel and len(year_indices) > 0:
        try:
            # Get date range for this year (with buffer for context)
            min_idx = min(year_indices)
            max_idx = max(year_indices)
            buffer_days = context_length * 2 + 30

            start_date = weekly_dates[max(0, min_idx - 10)].date() - timedelta(
                days=buffer_days
            )
            end_date = weekly_dates[min(max_idx, len(weekly_dates) - 1)].date()

            # Load daily OHLCV only
            daily_ohlcv = _load_daily_ohlcv(symbol, start_date, end_date)

            if daily_ohlcv is not None:
                logger.debug(
                    f"[WalkForward] Loaded {len(daily_ohlcv)} days of OHLCV for {symbol}"
                )
        except Exception as e:
            logger.warning(f"[WalkForward] Failed to load OHLCV data for {symbol}: {e}")
            use_multichannel = False

    model.eval()

    with torch.no_grad():
        for i in year_indices:
            ret, vol = _predict_single_week_patchtst(
                model=model,
                scaler=scaler,
                config=config,
                weekly_idx=i,
                weekly_prices=prices,
                weekly_dates=weekly_dates,
                daily_ohlcv=daily_ohlcv,
                daily_news=None,
                daily_fundamentals=None,
                use_multichannel=use_multichannel,
            )
            predictions.append(ret)
            volatilities.append(vol)

    return (predictions, volatilities)


def _predict_single_week_patchtst(
    model,
    scaler,
    config,
    weekly_idx: int,
    weekly_prices: np.ndarray,
    weekly_dates: pd.DatetimeIndex | None,
    daily_ohlcv: pd.DataFrame | None,
    daily_news: pd.DataFrame | None,
    daily_fundamentals: pd.DataFrame | None,
    use_multichannel: bool,
) -> tuple[float, float]:
    """Generate PatchTST weekly prediction using direct 5-day forecasting.

    Single forward pass produces (1, 5, 5) output -- 5 days x 5 channels.
    RevIN denormalizes output to original log-return scale automatically.
    Extract close_ret channel. NO scaler inverse-transform needed.

    Bug fixes applied:
    - Bug #E: Removed .permute(0, 2, 1) that crashed PatchTST
    - Bug #F: Uses exp(sum(log_returns)) - 1 instead of prod(1 + log_returns) - 1

    Returns:
        Tuple of (weekly_return, volatility) where volatility is std of daily returns.
    """
    import torch

    from brain_api.core.features import compute_ohlcv_log_returns

    context_length = config.context_length

    # Momentum fallback helper - returns (return, 0.0 volatility) since no daily path
    def momentum_fallback() -> tuple[float, float]:
        lookback = 4
        if weekly_idx >= lookback and weekly_prices[weekly_idx - lookback] > 0:
            ret = (
                weekly_prices[weekly_idx] - weekly_prices[weekly_idx - lookback]
            ) / weekly_prices[weekly_idx - lookback]
            return (ret, 0.0)
        return (0.0, 0.0)

    # Use 5-channel OHLCV features with single forward pass
    if use_multichannel and daily_ohlcv is not None and weekly_dates is not None:
        try:
            # Get cutoff date (Friday of this week - predict for next week)
            week_date = weekly_dates[weekly_idx]
            cutoff = week_date.date()

            # Filter data up to cutoff
            if daily_ohlcv.index.tz is not None:
                cutoff_ts = pd.Timestamp(cutoff).tz_localize(daily_ohlcv.index.tz)
            else:
                cutoff_ts = pd.Timestamp(cutoff)

            ohlcv_subset = daily_ohlcv[daily_ohlcv.index < cutoff_ts].copy()

            if len(ohlcv_subset) < context_length:
                return momentum_fallback()

            # Compute OHLCV log returns (5 channels)
            features_df = compute_ohlcv_log_returns(
                ohlcv_subset, use_returns=config.use_returns
            )
            if features_df.index.tz is not None:
                features_df.index = features_df.index.tz_localize(None)

            if len(features_df) < context_length:
                return momentum_fallback()

            # Extract only 5 OHLCV columns
            ohlcv_cols = ["open_ret", "high_ret", "low_ret", "close_ret", "volume_ret"]
            ohlcv_features = (
                features_df[ohlcv_cols].iloc[-context_length:].values
            )  # (60, 5)

            # close_ret channel index
            close_ret_idx = 3

            # NO scaler transform -- RevIN normalizes internally
            # Single forward pass -- NO permute! (Bug #E fix)
            x = torch.tensor(ohlcv_features, dtype=torch.float32).unsqueeze(
                0
            )  # (1, 60, 5)
            output = model(past_values=x).prediction_outputs  # (1, 5, 5) ORIGINAL scale

            # Extract close_ret channel -- already denormalized by RevIN
            daily_log_returns = output[0, :, close_ret_idx].cpu().numpy()  # (5,)

            # NO inverse-transform needed! RevIN already denormalized
            # Bug #F fix: exp(sum) not prod(1+)
            weekly_return = float(np.exp(np.sum(daily_log_returns)) - 1)
            volatility = float(np.std(daily_log_returns))
            return (weekly_return, volatility)

        except Exception as e:
            logger.debug(f"[WalkForward] PatchTST 5-channel inference failed: {e}")
            return momentum_fallback()

    # Fallback: momentum proxy (no daily OHLCV available)
    return momentum_fallback()


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


def _load_historical_news_for_walkforward(
    symbol: str, start_date: date, end_date: date
) -> pd.DataFrame | None:
    """Load historical news sentiment for walk-forward inference."""
    try:
        from brain_api.core.portfolio_rl.data_loading import (
            get_default_data_path,
            load_historical_news_sentiment,
        )

        news = load_historical_news_sentiment(
            [symbol],
            start_date,
            end_date,
            parquet_path=get_default_data_path() / "output" / "daily_sentiment.parquet",
        )
        return news.get(symbol)
    except Exception as e:
        logger.debug(f"[WalkForward] Failed to load news for {symbol}: {e}")
        return None


def _load_historical_fundamentals_for_walkforward(
    symbol: str, start_date: date, end_date: date
) -> pd.DataFrame | None:
    """Load historical fundamentals for walk-forward inference."""
    try:
        from brain_api.core.portfolio_rl.data_loading import (
            get_default_data_path,
            load_historical_fundamentals,
        )

        fund = load_historical_fundamentals(
            [symbol], start_date, end_date, cache_path=get_default_data_path()
        )
        return fund.get(symbol)
    except Exception as e:
        logger.debug(f"[WalkForward] Failed to load fundamentals for {symbol}: {e}")
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
    use_model_snapshots: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Build forecast features for RL training.

    Main entry point for generating walk-forward forecasts.

    Args:
        weekly_prices: Dict of symbol -> weekly price array
        weekly_dates: DatetimeIndex of weekly dates
        symbols: List of symbols
        forecaster_type: Which forecaster to use
        use_model_snapshots: Whether to use pre-trained model snapshots

    Returns:
        Tuple of (forecasts, volatilities) where each is
        Dict of symbol -> array of values
    """
    print(f"[PortfolioRL] Generating walk-forward forecasts ({forecaster_type})...")

    if use_model_snapshots:
        forecasts, volatilities = generate_walkforward_forecasts_with_model(
            weekly_prices, weekly_dates, symbols, forecaster_type
        )
    else:
        # Use momentum proxy (simple, no snapshots needed)
        forecasts, volatilities = generate_walkforward_forecasts_simple(
            weekly_prices, weekly_dates, symbols
        )

    print(f"[PortfolioRL] Generated forecasts for {len(forecasts)} symbols")
    return (forecasts, volatilities)


def build_dual_forecast_features(
    weekly_prices: dict[str, np.ndarray],
    weekly_dates: pd.DatetimeIndex,
    symbols: list[str],
    use_lstm_snapshots: bool = False,
    use_patchtst_snapshots: bool = False,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    """Build both LSTM and PatchTST forecast features for RL training.

    This is the main entry point for generating dual walk-forward forecasts
    for unified RL agents (PPO, SAC) that use both forecasters.

    Args:
        weekly_prices: Dict of symbol -> weekly price array
        weekly_dates: DatetimeIndex of weekly dates
        symbols: List of symbols
        use_lstm_snapshots: Whether to use pre-trained LSTM model snapshots
        use_patchtst_snapshots: Whether to use pre-trained PatchTST model snapshots

    Returns:
        Tuple of (lstm_forecasts, lstm_volatilities, patchtst_forecasts, patchtst_volatilities)
        where each is Dict of symbol -> array of values
    """
    print("[PortfolioRL] Generating dual walk-forward forecasts (LSTM + PatchTST)...")

    lstm_forecasts, lstm_volatilities = build_forecast_features(
        weekly_prices,
        weekly_dates,
        symbols,
        forecaster_type="lstm",
        use_model_snapshots=use_lstm_snapshots,
    )

    patchtst_forecasts, patchtst_volatilities = build_forecast_features(
        weekly_prices,
        weekly_dates,
        symbols,
        forecaster_type="patchtst",
        use_model_snapshots=use_patchtst_snapshots,
    )

    print(
        f"[PortfolioRL] Generated dual forecasts for {len(lstm_forecasts)} symbols "
        f"(LSTM: {use_lstm_snapshots}, PatchTST: {use_patchtst_snapshots})"
    )
    return (
        lstm_forecasts,
        lstm_volatilities,
        patchtst_forecasts,
        patchtst_volatilities,
    )

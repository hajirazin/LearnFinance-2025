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
) -> dict[str, np.ndarray]:
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
        Dict of symbol -> array of forecast values (same length as prices - 1)
    """
    forecasts: dict[str, np.ndarray] = {}

    if len(weekly_dates) == 0:
        return forecasts

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

    return forecasts


def generate_walkforward_forecasts_with_model(
    weekly_prices: dict[str, np.ndarray],
    weekly_dates: pd.DatetimeIndex,
    symbols: list[str],
    forecaster_type: Literal["lstm", "patchtst"],
    bootstrap_years: int = 4,
    snapshot_dir: Path | None = None,
) -> dict[str, np.ndarray]:
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
        Dict of symbol -> array of forecast values
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

    for symbol in symbols:
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

            if year < start_year + bootstrap_years:
                # Bootstrap: use momentum
                for i in year_indices:
                    if i < n_weeks - 1:
                        lookback = 4
                        if i >= lookback and prices[i - lookback] > 0:
                            symbol_forecasts[i] = (
                                prices[i] - prices[i - lookback]
                            ) / prices[i - lookback]
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
                        preds = _run_snapshot_inference(
                            snapshot_path,
                            forecaster_type,
                            symbol,
                            prices,
                            year_indices,
                            weekly_dates=weekly_dates,
                        )
                        for idx, pred in zip(year_indices, preds, strict=False):
                            if idx < n_weeks - 1:
                                symbol_forecasts[idx] = pred
                    except Exception as e:
                        print(
                            f"[WalkForward] Error running snapshot for {symbol} year {year}: {e}"
                        )
                        # Fallback to momentum
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

    return forecasts


def _run_snapshot_inference(
    snapshot_path: Path,
    forecaster_type: str,
    symbol: str,
    prices: np.ndarray,
    year_indices: list[int],
    weekly_dates: pd.DatetimeIndex | None = None,
) -> list[float]:
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
        List of predictions for each week in year_indices
    """
    from brain_api.storage.forecaster_snapshots import SnapshotLocalStorage

    # Load snapshot artifacts
    cutoff_date_str = snapshot_path.name.replace("snapshot_", "")
    cutoff_date = date.fromisoformat(cutoff_date_str)

    storage = SnapshotLocalStorage(forecaster_type)
    artifacts = storage.load_snapshot(cutoff_date)

    predictions = []

    if forecaster_type == "lstm":
        # Run LSTM inference (with multi-channel OHLCV support)
        predictions = _run_lstm_snapshot_inference(
            artifacts,
            prices,
            year_indices,
            weekly_dates=weekly_dates,
            symbol=symbol,
        )
    else:
        # Run PatchTST inference (multi-channel when available)
        predictions = _run_patchtst_snapshot_inference(
            artifacts,
            prices,
            year_indices,
            weekly_dates=weekly_dates,
            symbol=symbol,
        )

    return predictions


def _run_lstm_snapshot_inference(
    artifacts: "LSTMSnapshotArtifacts",
    prices: np.ndarray,
    year_indices: list[int],
    weekly_dates: pd.DatetimeIndex | None = None,
    symbol: str | None = None,
) -> list[float]:
    """Run LSTM snapshot inference for a symbol with multi-channel OHLCV support.

    Enhanced to load daily OHLCV data for proper 5-feature LSTM inference
    when available. Falls back to momentum if daily data unavailable.

    Args:
        artifacts: Loaded LSTM snapshot artifacts
        prices: Weekly price array for the symbol (fallback only)
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

    # Momentum fallback helper
    def momentum_fallback(idx: int) -> float:
        lookback = 4
        if idx >= lookback and prices[idx - lookback] > 0:
            return (prices[idx] - prices[idx - lookback]) / prices[idx - lookback]
        return 0.0

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
            predictions.append(momentum_fallback(i))
        return predictions

    model.eval()

    with torch.no_grad():
        for i in year_indices:
            pred = _predict_single_week_lstm(
                model=model,
                scaler=scaler,
                config=config,
                weekly_idx=i,
                weekly_prices=prices,
                weekly_dates=weekly_dates,
                daily_ohlcv=daily_ohlcv,
            )
            predictions.append(pred)

    return predictions


def _predict_single_week_lstm(
    model,
    scaler,
    config,
    weekly_idx: int,
    weekly_prices: np.ndarray,
    weekly_dates: pd.DatetimeIndex,
    daily_ohlcv: pd.DataFrame,
) -> float:
    """Generate LSTM prediction for a single week using 5-feature OHLCV.

    Uses daily OHLCV data to build proper 5-feature sequences matching
    how the LSTM was trained.
    """
    import torch

    from brain_api.core.features import compute_ohlcv_log_returns

    seq_len = config.sequence_length

    # Momentum fallback helper
    def momentum_fallback() -> float:
        lookback = 4
        if weekly_idx >= lookback and weekly_prices[weekly_idx - lookback] > 0:
            return (
                weekly_prices[weekly_idx] - weekly_prices[weekly_idx - lookback]
            ) / weekly_prices[weekly_idx - lookback]
        return 0.0

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

        # Take last seq_len rows
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
        if scaler is not None:
            features = scaler.transform(features)

        # Shape: (1, seq_len, 5)
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        # Run inference
        output = model(x)

        # Get prediction (expected weekly return)
        pred = output.squeeze().item()
        return pred

    except Exception as e:
        logger.debug(f"[WalkForward] LSTM inference failed: {e}")
        return momentum_fallback()


def _run_patchtst_snapshot_inference(
    artifacts: "PatchTSTSnapshotArtifacts",
    prices: np.ndarray,
    year_indices: list[int],
    weekly_dates: pd.DatetimeIndex | None = None,
    symbol: str | None = None,
) -> list[float]:
    """Run PatchTST snapshot inference for a symbol with multi-channel support.

    Enhanced to load daily OHLCV, news sentiment, and fundamentals for
    proper multi-channel PatchTST inference when available.

    Args:
        artifacts: Loaded PatchTST snapshot artifacts
        prices: Weekly price array for the symbol (fallback only)
        year_indices: Indices to predict
        weekly_dates: DatetimeIndex of weekly dates (for loading daily data)
        symbol: Stock symbol (for loading historical signals)

    Returns:
        List of predictions
    """
    import torch

    predictions = []
    model = artifacts.model
    scaler = artifacts.feature_scaler
    config = artifacts.config
    context_length = config.context_length

    # Determine if we can use multi-channel features
    use_multichannel = (
        weekly_dates is not None
        and symbol is not None
        and config.num_input_channels > 1
    )

    # Load daily data if multi-channel is enabled
    daily_ohlcv: pd.DataFrame | None = None
    daily_news: pd.DataFrame | None = None
    daily_fundamentals: pd.DataFrame | None = None

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

            # Load daily OHLCV
            daily_ohlcv = _load_daily_ohlcv(symbol, start_date, end_date)

            # Load news sentiment
            daily_news = _load_historical_news_for_walkforward(
                symbol, start_date, end_date
            )

            # Load fundamentals
            daily_fundamentals = _load_historical_fundamentals_for_walkforward(
                symbol, start_date, end_date
            )

            if daily_ohlcv is not None:
                logger.debug(
                    f"[WalkForward] Loaded {len(daily_ohlcv)} days of OHLCV for {symbol}"
                )
        except Exception as e:
            logger.warning(
                f"[WalkForward] Failed to load multi-channel data for {symbol}: {e}"
            )
            use_multichannel = False

    model.eval()

    with torch.no_grad():
        for i in year_indices:
            pred = _predict_single_week_patchtst(
                model=model,
                scaler=scaler,
                config=config,
                weekly_idx=i,
                weekly_prices=prices,
                weekly_dates=weekly_dates,
                daily_ohlcv=daily_ohlcv,
                daily_news=daily_news,
                daily_fundamentals=daily_fundamentals,
                use_multichannel=use_multichannel,
            )
            predictions.append(pred)

    return predictions


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
) -> float:
    """Generate PatchTST prediction for a single week.

    Uses multi-channel features if available, falls back to simple close returns.
    """
    import torch

    from brain_api.core.features import compute_ohlcv_log_returns

    context_length = config.context_length

    # Momentum fallback helper
    def momentum_fallback() -> float:
        lookback = 4
        if weekly_idx >= lookback and weekly_prices[weekly_idx - lookback] > 0:
            return (
                weekly_prices[weekly_idx] - weekly_prices[weekly_idx - lookback]
            ) / weekly_prices[weekly_idx - lookback]
        return 0.0

    # Try multi-channel approach
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

            # Take last context_length rows
            features_df = features_df.iloc[-context_length:]

            # Add news sentiment channel if available
            if daily_news is not None and len(daily_news) > 0:
                news_aligned = daily_news.reindex(
                    features_df.index, method="ffill"
                ).fillna(0.0)
                features_df["news_sentiment"] = news_aligned["sentiment_score"]
            elif config.num_input_channels > 5:
                features_df["news_sentiment"] = 0.0

            # Add fundamentals channels if available
            fund_cols = [
                "gross_margin",
                "operating_margin",
                "net_margin",
                "current_ratio",
                "debt_to_equity",
            ]
            if daily_fundamentals is not None and len(daily_fundamentals) > 0:
                fund_aligned = daily_fundamentals.reindex(
                    features_df.index, method="ffill"
                ).fillna(0.0)
                for col in fund_cols:
                    if col in fund_aligned.columns:
                        features_df[col] = fund_aligned[col]
                    elif config.num_input_channels > 6:
                        features_df[col] = 0.0
            elif config.num_input_channels > 6:
                for col in fund_cols:
                    features_df[col] = 0.0

            # Build feature array
            features = features_df.values  # (context_length, n_channels)

            # Match expected channels
            if features.shape[1] != config.num_input_channels:
                # Pad or truncate to match model
                if features.shape[1] < config.num_input_channels:
                    padding = np.zeros(
                        (
                            features.shape[0],
                            config.num_input_channels - features.shape[1],
                        )
                    )
                    features = np.hstack([features, padding])
                else:
                    features = features[:, : config.num_input_channels]

            # Apply scaler
            if scaler is not None:
                features = scaler.transform(features)

            # Shape: (1, context_length, n_channels) -> permute to (1, n_channels, context_length)
            x = (
                torch.tensor(features, dtype=torch.float32)
                .unsqueeze(0)
                .permute(0, 2, 1)
            )

            # Run inference
            output = model(past_values=x)

            # Get prediction
            if hasattr(output, "prediction_outputs"):
                scaled_pred = output.prediction_outputs[:, 0, 0].item()
            elif hasattr(output, "last_hidden_state"):
                scaled_pred = output.last_hidden_state[:, -1, 0].item()
            else:
                return momentum_fallback()

            # Inverse transform (close return is index 0)
            if scaler is not None:
                pred = scaled_pred * scaler.scale_[0] + scaler.mean_[0]
            else:
                pred = scaled_pred

            return pred

        except Exception as e:
            logger.debug(f"[WalkForward] Multi-channel inference failed: {e}")
            return momentum_fallback()

    # Fallback: single-channel from weekly prices
    if weekly_idx < context_length:
        return momentum_fallback()

    price_seq = weekly_prices[weekly_idx - context_length : weekly_idx]
    returns = np.diff(price_seq) / (price_seq[:-1] + 1e-8)

    if len(returns) != context_length - 1:
        return 0.0

    features = returns.reshape(1, -1, 1)

    if scaler is not None:
        flat_features = features.reshape(-1, features.shape[-1])
        flat_features = scaler.transform(flat_features)
        features = flat_features.reshape(features.shape)

    import torch

    x = torch.tensor(features, dtype=torch.float32).permute(0, 2, 1)

    try:
        output = model(past_values=x)
        if hasattr(output, "prediction_outputs"):
            scaled_pred = output.prediction_outputs[:, 0, 0].item()
        elif hasattr(output, "last_hidden_state"):
            scaled_pred = output.last_hidden_state[:, -1, 0].item()
        else:
            return momentum_fallback()

        if scaler is not None:
            pred = scaled_pred * scaler.scale_[0] + scaler.mean_[0]
        else:
            pred = scaled_pred
        return pred
    except Exception:
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
) -> dict[str, np.ndarray]:
    """Build forecast features for RL training.

    Main entry point for generating walk-forward forecasts.

    Args:
        weekly_prices: Dict of symbol -> weekly price array
        weekly_dates: DatetimeIndex of weekly dates
        symbols: List of symbols
        forecaster_type: Which forecaster to use
        use_model_snapshots: Whether to use pre-trained model snapshots

    Returns:
        Dict of symbol -> array of forecast values
    """
    print(f"[PortfolioRL] Generating walk-forward forecasts ({forecaster_type})...")

    if use_model_snapshots:
        forecasts = generate_walkforward_forecasts_with_model(
            weekly_prices, weekly_dates, symbols, forecaster_type
        )
    else:
        # Use momentum proxy (simple, no snapshots needed)
        forecasts = generate_walkforward_forecasts_simple(
            weekly_prices, weekly_dates, symbols
        )

    print(f"[PortfolioRL] Generated forecasts for {len(forecasts)} symbols")
    return forecasts

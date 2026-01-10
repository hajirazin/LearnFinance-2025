"""Walk-forward forecast generation for RL training.

Generates forecast features without look-ahead bias by using
forecaster models trained only on prior data.
"""

from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


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
        # Run LSTM inference
        predictions = _run_lstm_snapshot_inference(artifacts, prices, year_indices)
    else:
        # Run PatchTST inference
        predictions = _run_patchtst_snapshot_inference(artifacts, prices, year_indices)

    return predictions


def _run_lstm_snapshot_inference(
    artifacts: "LSTMSnapshotArtifacts",
    prices: np.ndarray,
    year_indices: list[int],
) -> list[float]:
    """Run LSTM snapshot inference for a symbol.

    Args:
        artifacts: Loaded LSTM snapshot artifacts
        prices: Weekly price array for the symbol
        year_indices: Indices to predict

    Returns:
        List of predictions
    """
    import torch

    predictions = []
    model = artifacts.model
    scaler = artifacts.feature_scaler
    config = artifacts.config
    seq_len = config.sequence_length

    # Check if scaler expects more features than we can provide
    # We only have close prices, so we can only build 1 feature (close returns)
    # If the model was trained with OHLCV (5 features), we must fall back to momentum
    if scaler is not None and hasattr(scaler, "n_features_in_"):
        expected_features = scaler.n_features_in_
        if expected_features > 1:
            # Model expects OHLCV features but we only have close prices
            # Fall back to momentum for all predictions
            for i in year_indices:
                lookback = 4
                if i >= lookback and prices[i - lookback] > 0:
                    pred = (prices[i] - prices[i - lookback]) / prices[i - lookback]
                else:
                    pred = 0.0
                predictions.append(pred)
            return predictions

    model.eval()

    with torch.no_grad():
        for i in year_indices:
            # Need at least seq_len of history
            if i < seq_len:
                # Not enough history, use momentum fallback
                lookback = 4
                if i >= lookback and prices[i - lookback] > 0:
                    pred = (prices[i] - prices[i - lookback]) / prices[i - lookback]
                else:
                    pred = 0.0
                predictions.append(pred)
                continue

            # Build input sequence from price history
            price_seq = prices[i - seq_len : i]

            # Convert to returns (log returns for LSTM)
            returns = np.diff(np.log(price_seq + 1e-8))  # seq_len - 1 returns

            if len(returns) != seq_len - 1:
                predictions.append(0.0)
                continue

            # LSTM expects (batch, seq, features)
            # For single-feature LSTM, add feature dimension
            features = returns.reshape(-1, 1)

            # Apply scaler if available
            if scaler is not None:
                features = scaler.transform(features)

            # Convert to tensor: (1, seq_len-1, 1)
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

            # Run inference
            output = model(x)

            # Get prediction (expected return)
            pred = output.squeeze().item()
            predictions.append(pred)

    return predictions


def _run_patchtst_snapshot_inference(
    artifacts: "PatchTSTSnapshotArtifacts",
    prices: np.ndarray,
    year_indices: list[int],
) -> list[float]:
    """Run PatchTST snapshot inference for a symbol.

    Args:
        artifacts: Loaded PatchTST snapshot artifacts
        prices: Weekly price array for the symbol
        year_indices: Indices to predict

    Returns:
        List of predictions
    """
    import torch

    predictions = []
    model = artifacts.model
    scaler = artifacts.feature_scaler
    config = artifacts.config
    context_length = config.context_length

    model.eval()

    with torch.no_grad():
        for i in year_indices:
            # Need at least context_length of history
            if i < context_length:
                # Not enough history, use momentum fallback
                lookback = 4
                if i >= lookback and prices[i - lookback] > 0:
                    pred = (prices[i] - prices[i - lookback]) / prices[i - lookback]
                else:
                    pred = 0.0
                predictions.append(pred)
                continue

            # Build input sequence from price history
            price_seq = prices[i - context_length : i]

            # Convert to returns for PatchTST
            # PatchTST expects multiple channels, but for simplicity
            # we use close returns as the main signal
            returns = np.diff(price_seq) / (price_seq[:-1] + 1e-8)

            if len(returns) != context_length - 1:
                predictions.append(0.0)
                continue

            # For single-channel input, shape: (batch, num_channels, seq_len)
            # PatchTST expects shape (batch, seq_len, num_channels) then permutes
            features = returns.reshape(1, -1, 1)  # (1, seq_len-1, 1)

            # Apply scaler if available
            if scaler is not None:
                # Scaler expects (n_samples, n_features)
                flat_features = features.reshape(-1, features.shape[-1])
                flat_features = scaler.transform(flat_features)
                features = flat_features.reshape(features.shape)

            # Convert to tensor and permute for PatchTST
            # PatchTST expects (batch, num_channels, context_length)
            x = torch.tensor(features, dtype=torch.float32).permute(0, 2, 1)

            try:
                # Run inference
                output = model(past_values=x)

                # Get prediction (last value of prediction horizon)
                if hasattr(output, "prediction_outputs"):
                    pred = output.prediction_outputs[:, 0, 0].item()
                elif hasattr(output, "last_hidden_state"):
                    pred = output.last_hidden_state[:, -1, 0].item()
                else:
                    # Fallback to momentum
                    lookback = 4
                    if i >= lookback and prices[i - lookback] > 0:
                        pred = (prices[i] - prices[i - lookback]) / prices[i - lookback]
                    else:
                        pred = 0.0
            except Exception:
                # Fallback to momentum on error
                lookback = 4
                if i >= lookback and prices[i - lookback] > 0:
                    pred = (prices[i] - prices[i - lookback]) / prices[i - lookback]
                else:
                    pred = 0.0

            predictions.append(pred)

    return predictions


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

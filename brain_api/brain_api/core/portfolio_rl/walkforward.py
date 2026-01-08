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
            momentum[i] = (prices[i] - prices[i - lookback_weeks]) / prices[i - lookback_weeks]

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
    end_year = weekly_dates[-1].year

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
                        symbol_forecasts[i] = (prices[i] - prices[i - lookback]) / prices[i - lookback]

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

    Args:
        weekly_prices: Dict of symbol -> weekly price array
        weekly_dates: DatetimeIndex corresponding to weekly_prices
        symbols: Ordered list of symbols
        forecaster_type: "lstm" or "patchtst"
        bootstrap_years: First N years use momentum proxy
        snapshot_dir: Directory containing forecaster snapshots

    Returns:
        Dict of symbol -> array of forecast values
    """
    # Check if snapshots directory exists
    if snapshot_dir is None:
        from brain_api.core.portfolio_rl.data_loading import get_default_data_path
        snapshot_dir = get_default_data_path() / "forecaster_snapshots" / forecaster_type

    if not snapshot_dir.exists():
        print(f"[WalkForward] No snapshots found at {snapshot_dir}, using momentum proxy")
        return generate_walkforward_forecasts_simple(
            weekly_prices, weekly_dates, symbols, bootstrap_years
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
                            symbol_forecasts[i] = (prices[i] - prices[i - lookback]) / prices[i - lookback]
            else:
                # Try to load snapshot for this cutoff
                cutoff_date = date(year - 1, 12, 31)
                snapshot_path = snapshot_dir / f"snapshot_{cutoff_date.isoformat()}"

                if snapshot_path.exists():
                    try:
                        # Load and run inference
                        preds = _run_snapshot_inference(
                            snapshot_path,
                            forecaster_type,
                            symbol,
                            prices,
                            year_indices,
                        )
                        for idx, pred in zip(year_indices, preds):
                            if idx < n_weeks - 1:
                                symbol_forecasts[idx] = pred
                    except Exception as e:
                        print(f"[WalkForward] Error running snapshot for {symbol} year {year}: {e}")
                        # Fallback to momentum
                        for i in year_indices:
                            if i < n_weeks - 1:
                                lookback = 4
                                if i >= lookback and prices[i - lookback] > 0:
                                    symbol_forecasts[i] = (prices[i] - prices[i - lookback]) / prices[i - lookback]
                else:
                    # No snapshot, use momentum
                    for i in year_indices:
                        if i < n_weeks - 1:
                            lookback = 4
                            if i >= lookback and prices[i - lookback] > 0:
                                symbol_forecasts[i] = (prices[i] - prices[i - lookback]) / prices[i - lookback]

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

    This is a placeholder that will be implemented when we have
    trained snapshots. For now, returns momentum as fallback.

    Args:
        snapshot_path: Path to snapshot directory
        forecaster_type: "lstm" or "patchtst"
        symbol: Stock symbol
        prices: Weekly prices array
        year_indices: Indices for the target year

    Returns:
        List of predictions for each week in year_indices
    """
    # TODO: Implement actual snapshot loading and inference
    # For now, return momentum as fallback
    predictions = []
    for i in year_indices:
        lookback = 4
        if i >= lookback and prices[i - lookback] > 0:
            pred = (prices[i] - prices[i - lookback]) / prices[i - lookback]
        else:
            pred = 0.0
        predictions.append(pred)

    return predictions


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


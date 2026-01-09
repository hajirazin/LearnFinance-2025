"""Training data structures for PPO + LSTM.

Contains data classes and functions for preparing training data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrainingData:
    """Prepared training data for PPO."""

    # Arrays aligned by week index
    symbol_returns: np.ndarray  # (n_weeks, n_stocks)
    signals: np.ndarray  # (n_weeks, n_stocks, n_signals)
    forecast_features: np.ndarray  # (n_weeks, n_stocks)

    # Metadata
    symbol_order: list[str]
    n_weeks: int
    n_stocks: int


def build_training_data(
    prices: dict[str, np.ndarray],
    signals: dict[str, dict[str, np.ndarray]],
    lstm_predictions: dict[str, np.ndarray],
    symbol_order: list[str],
) -> TrainingData:
    """Build training data arrays from raw data.

    Args:
        prices: Dict of symbol -> array of weekly prices.
        signals: Dict of symbol -> dict of signal_name -> array of daily values.
        lstm_predictions: Dict of symbol -> array of LSTM weekly return predictions.
        symbol_order: Ordered list of symbols to include.

    Returns:
        TrainingData with aligned arrays.
    """
    n_stocks = len(symbol_order)

    # Determine number of weeks from first symbol's prices
    first_symbol = symbol_order[0]
    n_weeks = len(prices[first_symbol]) - 1  # -1 because returns need two points

    # Signal names (must match StateSchema)
    signal_names = [
        "news_sentiment",
        "gross_margin",
        "operating_margin",
        "net_margin",
        "current_ratio",
        "debt_to_equity",
        "fundamental_age",
    ]
    n_signals = len(signal_names)

    # Build returns array
    symbol_returns = np.zeros((n_weeks, n_stocks))
    for stock_idx, symbol in enumerate(symbol_order):
        price_series = prices.get(symbol)
        if price_series is not None and len(price_series) > 1:
            # Weekly returns
            returns = (price_series[1:] - price_series[:-1]) / np.maximum(price_series[:-1], 1e-10)
            symbol_returns[:len(returns), stock_idx] = returns[:n_weeks]

    # Build signals array
    signals_array = np.zeros((n_weeks, n_stocks, n_signals))
    for stock_idx, symbol in enumerate(symbol_order):
        symbol_signals = signals.get(symbol, {})
        for signal_idx, signal_name in enumerate(signal_names):
            signal_values = symbol_signals.get(signal_name)
            if signal_values is not None:
                # Assume signals are weekly-aligned
                signals_array[:len(signal_values), stock_idx, signal_idx] = signal_values[:n_weeks]

    # Build forecast features array
    forecast_array = np.zeros((n_weeks, n_stocks))
    for stock_idx, symbol in enumerate(symbol_order):
        lstm_preds = lstm_predictions.get(symbol)
        if lstm_preds is not None:
            forecast_array[:len(lstm_preds), stock_idx] = lstm_preds[:n_weeks]

    return TrainingData(
        symbol_returns=symbol_returns,
        signals=signals_array,
        forecast_features=forecast_array,
        symbol_order=symbol_order,
        n_weeks=n_weeks,
        n_stocks=n_stocks,
    )


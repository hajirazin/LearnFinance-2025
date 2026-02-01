"""State building for portfolio RL.

The state vector contains:
- Market signals (news sentiment, fundamentals, fundamental_age)
- Forecast features (LSTM AND PatchTST predicted weekly returns)
- Current portfolio state (weights including CASH)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PortfolioState:
    """Current portfolio state for PPO decision-making.

    This is the "current_weights + cash" part of the state.
    """

    current_weights: dict[str, float]  # symbol -> weight, includes "CASH"
    cash_value: float  # absolute cash value (for reference)
    portfolio_value: float  # total portfolio value
    last_turnover: float = 0.0  # turnover from last rebalance

    def to_weight_array(self, symbol_order: list[str]) -> np.ndarray:
        """Convert to numpy array with CASH last.

        Args:
            symbol_order: Ordered list of stock symbols (not including CASH).

        Returns:
            Weights array with stocks first, CASH last.
        """
        n_assets = len(symbol_order) + 1
        weights = np.zeros(n_assets)

        for i, symbol in enumerate(symbol_order):
            weights[i] = self.current_weights.get(symbol, 0.0)

        weights[-1] = self.current_weights.get("CASH", 0.0)

        return weights


@dataclass
class StateSchema:
    """Schema defining the state vector structure.

    The state vector is a flat numpy array with the following segments:
    1. Per-stock signals (n_stocks * n_signals_per_stock)
    2. Per-stock LSTM forecast return (n_stocks * 1)
    3. Per-stock LSTM forecast volatility (n_stocks * 1)
    4. Per-stock PatchTST forecast return (n_stocks * 1)
    5. Per-stock PatchTST forecast volatility (n_stocks * 1)
    6. Current portfolio weights (n_stocks + 1 for CASH)

    Signals per stock:
    - news_sentiment (1)
    - gross_margin (1)
    - operating_margin (1)
    - net_margin (1)
    - current_ratio (1)
    - debt_to_equity (1)
    - fundamental_age (1)

    Total per stock = 7 signals + 4 forecasts (return+vol for each) = 11 features
    State dim for 15 stocks = 15*7 + 15*4 + 16 = 105 + 60 + 16 = 181
    """

    n_stocks: int = 15
    n_signals_per_stock: int = 7  # news + 5 fundamentals + fundamental_age
    n_forecasts_per_stock: int = (
        4  # LSTM return + LSTM vol + PatchTST return + PatchTST vol
    )

    @property
    def n_forecast_features(self) -> int:
        """Total forecast features (LSTM + PatchTST for each stock)."""
        return self.n_stocks * self.n_forecasts_per_stock

    @property
    def n_portfolio_weights(self) -> int:
        """Portfolio weights including CASH."""
        return self.n_stocks + 1

    @property
    def state_dim(self) -> int:
        """Total state vector dimension."""
        return (
            self.n_stocks * self.n_signals_per_stock  # signals
            + self.n_forecast_features  # forecasts (LSTM + PatchTST)
            + self.n_portfolio_weights  # portfolio
        )

    @property
    def signal_names(self) -> list[str]:
        """Names of per-stock signals."""
        return [
            "news_sentiment",
            "gross_margin",
            "operating_margin",
            "net_margin",
            "current_ratio",
            "debt_to_equity",
            "fundamental_age",
        ]

    def get_signal_indices(self, stock_idx: int) -> tuple[int, int]:
        """Get start/end indices for a stock's signals."""
        start = stock_idx * self.n_signals_per_stock
        end = start + self.n_signals_per_stock
        return start, end

    def get_forecast_indices(self) -> tuple[int, int]:
        """Get start/end indices for ALL forecast features (LSTM + PatchTST)."""
        start = self.n_stocks * self.n_signals_per_stock
        end = start + self.n_forecast_features
        return start, end

    def get_lstm_forecast_indices(self) -> tuple[int, int]:
        """Get start/end indices for LSTM forecast return features."""
        start = self.n_stocks * self.n_signals_per_stock
        end = start + self.n_stocks
        return start, end

    def get_lstm_volatility_indices(self) -> tuple[int, int]:
        """Get start/end indices for LSTM forecast volatility features."""
        start = self.n_stocks * self.n_signals_per_stock + self.n_stocks
        end = start + self.n_stocks
        return start, end

    def get_patchtst_forecast_indices(self) -> tuple[int, int]:
        """Get start/end indices for PatchTST forecast return features."""
        start = self.n_stocks * self.n_signals_per_stock + 2 * self.n_stocks
        end = start + self.n_stocks
        return start, end

    def get_patchtst_volatility_indices(self) -> tuple[int, int]:
        """Get start/end indices for PatchTST forecast volatility features."""
        start = self.n_stocks * self.n_signals_per_stock + 3 * self.n_stocks
        end = start + self.n_stocks
        return start, end

    def get_portfolio_indices(self) -> tuple[int, int]:
        """Get start/end indices for portfolio weights."""
        start = self.n_stocks * self.n_signals_per_stock + self.n_forecast_features
        end = start + self.n_portfolio_weights
        return start, end


def build_state_vector(
    signals: dict[str, dict[str, float]],
    lstm_forecasts: dict[str, float],
    patchtst_forecasts: dict[str, float],
    portfolio_weights: np.ndarray,
    symbol_order: list[str],
    schema: StateSchema | None = None,
    lstm_volatilities: dict[str, float] | None = None,
    patchtst_volatilities: dict[str, float] | None = None,
) -> np.ndarray:
    """Build the full state vector for RL agents (PPO/SAC).

    Args:
        signals: Dict of symbol -> signal_dict.
                 Each signal_dict has keys: news_sentiment, gross_margin,
                 operating_margin, net_margin, current_ratio, debt_to_equity,
                 fundamental_age.
        lstm_forecasts: Dict of symbol -> LSTM predicted weekly return.
        patchtst_forecasts: Dict of symbol -> PatchTST predicted weekly return.
        portfolio_weights: Current portfolio weights with CASH last.
        symbol_order: Ordered list of stock symbols (determines ordering).
        schema: State schema (created from defaults if None).
        lstm_volatilities: Dict of symbol -> LSTM forecast volatility (std of daily returns).
        patchtst_volatilities: Dict of symbol -> PatchTST forecast volatility.

    Returns:
        Flat state vector of shape (state_dim,).
    """
    if schema is None:
        schema = StateSchema(n_stocks=len(symbol_order))

    if lstm_volatilities is None:
        lstm_volatilities = {}
    if patchtst_volatilities is None:
        patchtst_volatilities = {}

    state = np.zeros(schema.state_dim)

    # 1. Fill per-stock signals
    signal_names = schema.signal_names
    for stock_idx, symbol in enumerate(symbol_order):
        start, _end = schema.get_signal_indices(stock_idx)
        symbol_signals = signals.get(symbol, {})

        for signal_idx, signal_name in enumerate(signal_names):
            state[start + signal_idx] = symbol_signals.get(signal_name, 0.0)

    # 2. Fill LSTM forecast return features
    lstm_start, _lstm_end = schema.get_lstm_forecast_indices()
    for stock_idx, symbol in enumerate(symbol_order):
        state[lstm_start + stock_idx] = lstm_forecasts.get(symbol, 0.0)

    # 3. Fill LSTM volatility features
    lstm_vol_start, _lstm_vol_end = schema.get_lstm_volatility_indices()
    for stock_idx, symbol in enumerate(symbol_order):
        state[lstm_vol_start + stock_idx] = lstm_volatilities.get(symbol, 0.0)

    # 4. Fill PatchTST forecast return features
    patchtst_start, _patchtst_end = schema.get_patchtst_forecast_indices()
    for stock_idx, symbol in enumerate(symbol_order):
        state[patchtst_start + stock_idx] = patchtst_forecasts.get(symbol, 0.0)

    # 5. Fill PatchTST volatility features
    patchtst_vol_start, _patchtst_vol_end = schema.get_patchtst_volatility_indices()
    for stock_idx, symbol in enumerate(symbol_order):
        state[patchtst_vol_start + stock_idx] = patchtst_volatilities.get(symbol, 0.0)

    # 6. Fill portfolio weights
    portfolio_start, portfolio_end = schema.get_portfolio_indices()
    state[portfolio_start:portfolio_end] = portfolio_weights

    return state


def extract_portfolio_weights_from_state(
    state: np.ndarray,
    schema: StateSchema,
) -> np.ndarray:
    """Extract portfolio weights from state vector.

    Args:
        state: Full state vector.
        schema: State schema.

    Returns:
        Portfolio weights with CASH last.
    """
    start, end = schema.get_portfolio_indices()
    return state[start:end].copy()


def state_to_dict(
    state: np.ndarray,
    symbol_order: list[str],
    schema: StateSchema | None = None,
) -> dict[str, Any]:
    """Convert state vector back to structured dict.

    Useful for serialization and debugging.

    Args:
        state: Flat state vector.
        symbol_order: Ordered list of stock symbols.
        schema: State schema.

    Returns:
        Structured dict with signals, forecasts, volatilities, and portfolio weights.
    """
    if schema is None:
        schema = StateSchema(n_stocks=len(symbol_order))

    result: dict[str, Any] = {
        "signals": {},
        "lstm_forecasts": {},
        "lstm_volatilities": {},
        "patchtst_forecasts": {},
        "patchtst_volatilities": {},
        "current_weights": {},
    }

    # Extract signals
    signal_names = schema.signal_names
    for stock_idx, symbol in enumerate(symbol_order):
        start, _end = schema.get_signal_indices(stock_idx)
        result["signals"][symbol] = {}
        for signal_idx, signal_name in enumerate(signal_names):
            result["signals"][symbol][signal_name] = float(state[start + signal_idx])

    # Extract LSTM forecast return features
    lstm_start, _lstm_end = schema.get_lstm_forecast_indices()
    for stock_idx, symbol in enumerate(symbol_order):
        result["lstm_forecasts"][symbol] = float(state[lstm_start + stock_idx])

    # Extract LSTM volatility features
    lstm_vol_start, _lstm_vol_end = schema.get_lstm_volatility_indices()
    for stock_idx, symbol in enumerate(symbol_order):
        result["lstm_volatilities"][symbol] = float(state[lstm_vol_start + stock_idx])

    # Extract PatchTST forecast return features
    patchtst_start, _patchtst_end = schema.get_patchtst_forecast_indices()
    for stock_idx, symbol in enumerate(symbol_order):
        result["patchtst_forecasts"][symbol] = float(state[patchtst_start + stock_idx])

    # Extract PatchTST volatility features
    patchtst_vol_start, _patchtst_vol_end = schema.get_patchtst_volatility_indices()
    for stock_idx, symbol in enumerate(symbol_order):
        result["patchtst_volatilities"][symbol] = float(
            state[patchtst_vol_start + stock_idx]
        )

    # Extract portfolio weights
    portfolio_start, portfolio_end = schema.get_portfolio_indices()
    weights = state[portfolio_start:portfolio_end]
    for stock_idx, symbol in enumerate(symbol_order):
        result["current_weights"][symbol] = float(weights[stock_idx])
    result["current_weights"]["CASH"] = float(weights[-1])

    return result

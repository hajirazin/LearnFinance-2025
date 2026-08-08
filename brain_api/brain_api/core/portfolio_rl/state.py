"""State building for portfolio RL.

The state vector contains:
- Market signals (news sentiment, coverage, fundamentals, fundamental_age)
- Forecast features (PatchTST predicted weekly returns only)
- Current portfolio state (weights including CASH)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PortfolioState:
    """Current portfolio state for RL decision-making.

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
    """Schema defining the live SAC state vector structure.

    Segments:
    1. Per-stock signals (n_stocks * n_signals_per_stock)
    2. Per-stock PatchTST forecast return (n_stocks * 1)
    3. Current portfolio weights (n_stocks + 1 for CASH)

    For n_stocks=15: 15*7 + 15*1 + 16 = 136.
    """

    n_stocks: int = 15
    n_forecasts_per_stock: int = 1  # PatchTST return only

    @property
    def n_signals_per_stock(self) -> int:
        """Number of per-stock signal features."""
        return len(self.signal_names)

    @property
    def n_forecast_features(self) -> int:
        """Total PatchTST forecast features."""
        return self.n_stocks * self.n_forecasts_per_stock

    @property
    def n_portfolio_weights(self) -> int:
        """Portfolio weights including CASH."""
        return self.n_stocks + 1

    @property
    def state_dim(self) -> int:
        """Total state vector dimension."""
        return (
            self.n_stocks * self.n_signals_per_stock
            + self.n_forecast_features
            + self.n_portfolio_weights
        )

    @property
    def signal_names(self) -> list[str]:
        """Names of per-stock signals."""
        from brain_api.core.sac.decision_context import SAC_SIGNAL_NAMES

        return list(SAC_SIGNAL_NAMES)

    def get_signal_indices(self, stock_idx: int) -> tuple[int, int]:
        """Get start/end indices for a stock's signals."""
        start = stock_idx * self.n_signals_per_stock
        end = start + self.n_signals_per_stock
        return start, end

    def get_forecast_indices(self) -> tuple[int, int]:
        """Get start/end indices for all PatchTST forecast features."""
        start = self.n_stocks * self.n_signals_per_stock
        end = start + self.n_forecast_features
        return start, end

    def get_patchtst_forecast_indices(self) -> tuple[int, int]:
        """Get start/end indices for PatchTST forecast return features."""
        return self.get_forecast_indices()

    def get_portfolio_indices(self) -> tuple[int, int]:
        """Get start/end indices for portfolio weights."""
        start = self.n_stocks * self.n_signals_per_stock + self.n_forecast_features
        end = start + self.n_portfolio_weights
        return start, end


def build_state_vector(
    signals: dict[str, dict[str, float]],
    patchtst_forecasts: dict[str, float],
    portfolio_weights: np.ndarray,
    symbol_order: list[str],
    schema: StateSchema | None = None,
) -> np.ndarray:
    """Build the full state vector for RL agents (SAC).

    Args:
        signals: Dict of symbol -> signal_dict (SAC_SIGNAL_NAMES keys).
        patchtst_forecasts: Dict of symbol -> PatchTST predicted weekly return.
        portfolio_weights: Current portfolio weights with CASH last.
        symbol_order: Ordered list of stock symbols (determines ordering).
        schema: State schema (created from defaults if None).

    Returns:
        Flat state vector of shape (state_dim,).
    """
    if schema is None:
        schema = StateSchema(n_stocks=len(symbol_order))

    if len(symbol_order) != schema.n_stocks:
        raise ValueError(
            f"symbol_order has {len(symbol_order)} stocks, schema expects "
            f"{schema.n_stocks}"
        )
    if portfolio_weights.shape != (schema.n_portfolio_weights,):
        raise ValueError(
            f"portfolio_weights shape {portfolio_weights.shape} does not match "
            f"({schema.n_portfolio_weights},)"
        )

    state = np.zeros(schema.state_dim)

    signal_names = schema.signal_names
    for stock_idx, symbol in enumerate(symbol_order):
        start, _end = schema.get_signal_indices(stock_idx)
        if symbol not in signals:
            raise ValueError(f"Missing required SAC signals for {symbol}")
        symbol_signals = signals[symbol]

        for signal_idx, signal_name in enumerate(signal_names):
            if signal_name not in symbol_signals:
                raise ValueError(
                    f"Missing required SAC signal {signal_name!r} for {symbol}"
                )
            value = symbol_signals[signal_name]
            if not np.isfinite(value):
                raise ValueError(
                    f"Non-finite SAC signal {signal_name!r} for {symbol}: {value}"
                )
            state[start + signal_idx] = value

    patchtst_start, _patchtst_end = schema.get_patchtst_forecast_indices()
    for stock_idx, symbol in enumerate(symbol_order):
        if symbol not in patchtst_forecasts:
            raise ValueError(f"Missing required PatchTST forecast for {symbol}")
        value = patchtst_forecasts[symbol]
        if not np.isfinite(value):
            raise ValueError(f"Non-finite PatchTST forecast for {symbol}: {value}")
        state[patchtst_start + stock_idx] = value

    if not np.all(np.isfinite(portfolio_weights)):
        raise ValueError("Portfolio weights must all be finite")
    if np.any(portfolio_weights < 0):
        raise ValueError("Portfolio weights must all be nonnegative")
    if not np.isclose(float(portfolio_weights.sum()), 1.0, atol=1e-8):
        raise ValueError(
            f"Portfolio weights must sum to 1.0, got {portfolio_weights.sum()}"
        )

    portfolio_start, portfolio_end = schema.get_portfolio_indices()
    state[portfolio_start:portfolio_end] = portfolio_weights

    return state


def extract_portfolio_weights_from_state(
    state: np.ndarray,
    schema: StateSchema,
) -> np.ndarray:
    """Extract portfolio weights from state vector."""
    start, end = schema.get_portfolio_indices()
    return state[start:end].copy()


def state_to_dict(
    state: np.ndarray,
    symbol_order: list[str],
    schema: StateSchema | None = None,
) -> dict[str, Any]:
    """Convert state vector back to structured dict."""
    if schema is None:
        schema = StateSchema(n_stocks=len(symbol_order))

    result: dict[str, Any] = {
        "signals": {},
        "patchtst_forecasts": {},
        "current_weights": {},
    }

    signal_names = schema.signal_names
    for stock_idx, symbol in enumerate(symbol_order):
        start, _end = schema.get_signal_indices(stock_idx)
        result["signals"][symbol] = {}
        for signal_idx, signal_name in enumerate(signal_names):
            result["signals"][symbol][signal_name] = float(state[start + signal_idx])

    patchtst_start, _patchtst_end = schema.get_patchtst_forecast_indices()
    for stock_idx, symbol in enumerate(symbol_order):
        result["patchtst_forecasts"][symbol] = float(state[patchtst_start + stock_idx])

    portfolio_start, portfolio_end = schema.get_portfolio_indices()
    weights = state[portfolio_start:portfolio_end]
    for stock_idx, symbol in enumerate(symbol_order):
        result["current_weights"][symbol] = float(weights[stock_idx])
    result["current_weights"]["CASH"] = float(weights[-1])

    return result

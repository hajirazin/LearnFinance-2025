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
    """Schema defining the state vector structure.

    The state vector is a flat numpy array with the following segments:
    1. Per-stock signals (n_stocks * n_signals_per_stock)
    2. Per-stock LSTM forecast return (n_stocks * 1)
    3. Per-stock PatchTST forecast return (n_stocks * 1)
    4. Current portfolio weights (n_stocks + 1 for CASH)

    Signals per stock:
    - news_sentiment (1)
    - gross_margin (1)
    - operating_margin (1)
    - net_margin (1)
    - current_ratio (1)
    - debt_to_equity (1)
    - fundamental_age (1)

    Total per stock = 7 signals + 2 forecasts (return only, no vol) = 9 features
    State dim for 15 stocks = 15*7 + 15*2 + 16 = 105 + 30 + 16 = 151
    """

    n_stocks: int = 15
    schema_version: int = 1
    n_forecasts_per_stock: int = 2  # LSTM return + PatchTST return (no volatility)

    def __post_init__(self) -> None:
        if self.schema_version not in (1, 2):
            raise ValueError(
                f"Unsupported SAC state schema version: {self.schema_version}"
            )

    @classmethod
    def v1(cls, n_stocks: int) -> StateSchema:
        """Construct the 151-at-15 schema used by legacy artifacts."""
        return cls(n_stocks=n_stocks, schema_version=1)

    @classmethod
    def v2(cls, n_stocks: int) -> StateSchema:
        """Construct the strict 166-at-15 schema used by new artifacts."""
        return cls(n_stocks=n_stocks, schema_version=2)

    @property
    def version(self) -> int:
        """Public artifact schema version."""
        return self.schema_version

    @property
    def n_signals_per_stock(self) -> int:
        """Number of signals in the selected artifact-compatible schema."""
        return len(self.signal_names)

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
        if self.schema_version >= 2:
            from brain_api.core.sac.decision_context import SAC_V2_SIGNAL_NAMES

            return list(SAC_V2_SIGNAL_NAMES)
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

    def get_patchtst_forecast_indices(self) -> tuple[int, int]:
        """Get start/end indices for PatchTST forecast return features."""
        start = self.n_stocks * self.n_signals_per_stock + 1 * self.n_stocks
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
) -> np.ndarray:
    """Build the full state vector for RL agents (SAC).

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

    strict = schema.schema_version >= 2
    state = np.zeros(schema.state_dim)

    # 1. Fill per-stock signals
    signal_names = schema.signal_names
    for stock_idx, symbol in enumerate(symbol_order):
        start, _end = schema.get_signal_indices(stock_idx)
        if strict and symbol not in signals:
            raise ValueError(f"Missing required SAC signals for {symbol}")
        symbol_signals = signals.get(symbol, {})

        for signal_idx, signal_name in enumerate(signal_names):
            if strict and signal_name not in symbol_signals:
                raise ValueError(
                    f"Missing required SAC signal {signal_name!r} for {symbol}"
                )
            value = symbol_signals.get(signal_name, 0.0)
            if strict and not np.isfinite(value):
                raise ValueError(
                    f"Non-finite SAC signal {signal_name!r} for {symbol}: {value}"
                )
            state[start + signal_idx] = value

    # 2. Fill LSTM forecast return features
    lstm_start, _lstm_end = schema.get_lstm_forecast_indices()
    for stock_idx, symbol in enumerate(symbol_order):
        if strict and symbol not in lstm_forecasts:
            raise ValueError(f"Missing required LSTM forecast for {symbol}")
        value = lstm_forecasts.get(symbol, 0.0)
        if strict and not np.isfinite(value):
            raise ValueError(f"Non-finite LSTM forecast for {symbol}: {value}")
        state[lstm_start + stock_idx] = value

    # 3. Fill PatchTST forecast return features
    patchtst_start, _patchtst_end = schema.get_patchtst_forecast_indices()
    for stock_idx, symbol in enumerate(symbol_order):
        if strict and symbol not in patchtst_forecasts:
            raise ValueError(f"Missing required PatchTST forecast for {symbol}")
        value = patchtst_forecasts.get(symbol, 0.0)
        if strict and not np.isfinite(value):
            raise ValueError(f"Non-finite PatchTST forecast for {symbol}: {value}")
        state[patchtst_start + stock_idx] = value

    if strict:
        if not np.all(np.isfinite(portfolio_weights)):
            raise ValueError("Portfolio weights must all be finite")
        if np.any(portfolio_weights < 0):
            raise ValueError("Portfolio weights must all be nonnegative")
        if not np.isclose(float(portfolio_weights.sum()), 1.0, atol=1e-8):
            raise ValueError(
                f"Portfolio weights must sum to 1.0, got {portfolio_weights.sum()}"
            )

    # 4. Fill portfolio weights
    portfolio_start, portfolio_end = schema.get_portfolio_indices()
    state[portfolio_start:portfolio_end] = portfolio_weights

    return state


def build_state_vector_strict_v2(
    signals: dict[str, dict[str, float]],
    lstm_forecasts: dict[str, float],
    patchtst_forecasts: dict[str, float],
    portfolio_weights: np.ndarray,
    symbol_order: list[str],
) -> np.ndarray:
    """Build a strict state-v2 vector, failing on incomplete actor inputs."""
    return build_state_vector(
        signals=signals,
        lstm_forecasts=lstm_forecasts,
        patchtst_forecasts=patchtst_forecasts,
        portfolio_weights=portfolio_weights,
        symbol_order=symbol_order,
        schema=StateSchema.v2(len(symbol_order)),
    )


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
        Structured dict with signals, forecasts, and portfolio weights.
    """
    if schema is None:
        schema = StateSchema(n_stocks=len(symbol_order))

    result: dict[str, Any] = {
        "signals": {},
        "lstm_forecasts": {},
        "patchtst_forecasts": {},
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

    # Extract PatchTST forecast return features
    patchtst_start, _patchtst_end = schema.get_patchtst_forecast_indices()
    for stock_idx, symbol in enumerate(symbol_order):
        result["patchtst_forecasts"][symbol] = float(state[patchtst_start + stock_idx])

    # Extract portfolio weights
    portfolio_start, portfolio_end = schema.get_portfolio_indices()
    weights = state[portfolio_start:portfolio_end]
    for stock_idx, symbol in enumerate(symbol_order):
        result["current_weights"][symbol] = float(weights[stock_idx])
    result["current_weights"]["CASH"] = float(weights[-1])

    return result

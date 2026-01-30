"""Weekly portfolio environment for PPO training.

This environment simulates weekly portfolio rebalancing with:
- Long-only simplex weights + CASH
- Transaction costs
- Constraint enforcement (cash buffer, max position)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from brain_api.core.portfolio_rl.config import DEFAULT_PPO_BASE_CONFIG, PPOBaseConfig
from brain_api.core.portfolio_rl.constraints import (
    apply_softmax_to_weights,
    compute_turnover,
    enforce_constraints,
)
from brain_api.core.portfolio_rl.rewards import compute_reward
from brain_api.core.portfolio_rl.state import (
    StateSchema,
    build_state_vector,
)


@dataclass
class EnvStep:
    """Result of one environment step."""

    next_state: np.ndarray
    reward: float
    done: bool
    info: dict[str, Any]


class PortfolioEnv:
    """Weekly portfolio rebalancing environment.

    Episode structure:
    - One episode = one year (52 weeks) or until data ends
    - Each step = one week
    - Action = raw logits for portfolio weights
    - State = signals + forecast + current portfolio weights

    The environment is stateful and tracks:
    - Current portfolio weights
    - Current week index within episode
    - Episode year
    """

    def __init__(
        self,
        symbol_returns: np.ndarray,
        signals: np.ndarray,
        lstm_forecasts: np.ndarray,
        patchtst_forecasts: np.ndarray,
        symbol_order: list[str],
        config: PPOBaseConfig | None = None,
    ):
        """Initialize environment.

        Args:
            symbol_returns: Weekly returns for each symbol.
                           Shape: (n_weeks, n_stocks).
            signals: Per-stock signals for each week.
                    Shape: (n_weeks, n_stocks, n_signals_per_stock).
            lstm_forecasts: LSTM forecast feature for each stock each week.
                           Shape: (n_weeks, n_stocks).
            patchtst_forecasts: PatchTST forecast feature for each stock each week.
                               Shape: (n_weeks, n_stocks).
            symbol_order: Ordered list of stock symbols.
            config: PPO configuration.
        """
        self.symbol_returns = symbol_returns
        self.signals = signals
        self.lstm_forecasts = lstm_forecasts
        self.patchtst_forecasts = patchtst_forecasts
        self.symbol_order = symbol_order
        self.config = config or DEFAULT_PPO_BASE_CONFIG

        self.n_weeks = symbol_returns.shape[0]
        self.n_stocks = len(symbol_order)

        # State schema
        self.schema = StateSchema(n_stocks=self.n_stocks)

        # Episode state
        self.current_week_idx: int = 0
        self.current_weights: np.ndarray = self._initial_weights()
        self.episode_start_week: int = 0

        # For tracking
        self.episode_returns: list[float] = []
        self.episode_turnovers: list[float] = []

    @property
    def state_dim(self) -> int:
        """Dimension of state vector."""
        return self.schema.state_dim

    @property
    def action_dim(self) -> int:
        """Dimension of action vector (n_stocks + 1 for CASH)."""
        return self.n_stocks + 1

    def _initial_weights(self) -> np.ndarray:
        """Get initial portfolio weights (100% CASH)."""
        weights = np.zeros(self.action_dim)
        weights[-1] = 1.0  # CASH is last
        return weights

    def _build_state(self, week_idx: int) -> np.ndarray:
        """Build state vector for a given week.

        Args:
            week_idx: Index of the week.

        Returns:
            State vector.
        """
        # Get signals for this week
        week_signals = self.signals[week_idx]  # (n_stocks, n_signals)
        signals_dict = {}
        signal_names = self.schema.signal_names
        for stock_idx, symbol in enumerate(self.symbol_order):
            signals_dict[symbol] = {}
            for signal_idx, signal_name in enumerate(signal_names):
                signals_dict[symbol][signal_name] = float(
                    week_signals[stock_idx, signal_idx]
                )

        # Get LSTM forecast features for this week
        week_lstm = self.lstm_forecasts[week_idx]  # (n_stocks,)
        lstm_dict = {
            symbol: float(week_lstm[stock_idx])
            for stock_idx, symbol in enumerate(self.symbol_order)
        }

        # Get PatchTST forecast features for this week
        week_patchtst = self.patchtst_forecasts[week_idx]  # (n_stocks,)
        patchtst_dict = {
            symbol: float(week_patchtst[stock_idx])
            for stock_idx, symbol in enumerate(self.symbol_order)
        }

        return build_state_vector(
            signals=signals_dict,
            lstm_forecasts=lstm_dict,
            patchtst_forecasts=patchtst_dict,
            portfolio_weights=self.current_weights,
            symbol_order=self.symbol_order,
            schema=self.schema,
        )

    def reset(self, start_week: int | None = None) -> np.ndarray:
        """Reset environment for a new episode.

        Args:
            start_week: Starting week index. If None, randomly sampled.

        Returns:
            Initial state vector.
        """
        # Determine start week
        if start_week is not None:
            self.episode_start_week = start_week
        else:
            # Random start, but leave room for at least 52 weeks
            max_start = max(0, self.n_weeks - 52)
            if max_start > 0:
                self.episode_start_week = np.random.randint(0, max_start)
            else:
                self.episode_start_week = 0

        self.current_week_idx = self.episode_start_week
        self.current_weights = self._initial_weights()
        self.episode_returns = []
        self.episode_turnovers = []

        return self._build_state(self.current_week_idx)

    def step(self, action: np.ndarray) -> EnvStep:
        """Take one step in the environment.

        Args:
            action: Raw logits for portfolio weights (n_stocks + 1).

        Returns:
            EnvStep with next_state, reward, done, info.
        """
        # Convert action to weights via softmax
        target_weights = apply_softmax_to_weights(action)

        # Enforce constraints
        target_weights = enforce_constraints(
            target_weights,
            cash_buffer=self.config.cash_buffer,
            max_position_weight=self.config.max_position_weight,
        )

        # Compute turnover
        turnover = compute_turnover(self.current_weights, target_weights)

        # Get weekly returns for stocks (CASH return = 0)
        stock_returns = self.symbol_returns[self.current_week_idx]  # (n_stocks,)
        asset_returns = np.zeros(self.action_dim)
        asset_returns[: self.n_stocks] = stock_returns
        # CASH return is 0 (could add risk-free rate if desired)

        # Compute portfolio return using target weights
        # (assumes rebalance happens at week start)
        portfolio_return = float(np.dot(target_weights, asset_returns))

        # Compute reward
        reward = compute_reward(portfolio_return, turnover, self.config)

        # Track for episode statistics
        self.episode_returns.append(portfolio_return)
        self.episode_turnovers.append(turnover)

        # Update state
        self.current_weights = target_weights
        self.current_week_idx += 1

        # Check if episode is done
        # Done if: (1) 52 weeks passed, or (2) end of data
        weeks_in_episode = self.current_week_idx - self.episode_start_week
        done = (
            weeks_in_episode >= 52  # ~1 year
            or self.current_week_idx >= self.n_weeks
        )

        # Build next state (if not done)
        if done:
            next_state = np.zeros(self.state_dim)  # Dummy state
        else:
            next_state = self._build_state(self.current_week_idx)

        info = {
            "portfolio_return": portfolio_return,
            "turnover": turnover,
            "target_weights": target_weights.tolist(),
            "week_idx": self.current_week_idx - 1,
        }

        return EnvStep(
            next_state=next_state,
            reward=reward,
            done=done,
            info=info,
        )

    def get_episode_metrics(self) -> dict[str, float]:
        """Get metrics for the current episode.

        Returns:
            Dict with episode statistics.
        """
        returns = np.array(self.episode_returns)
        turnovers = np.array(self.episode_turnovers)

        if len(returns) == 0:
            return {
                "episode_return": 0.0,
                "episode_sharpe": 0.0,
                "avg_turnover": 0.0,
                "n_weeks": 0,
            }

        # Cumulative return
        cumulative_return = float(np.prod(1 + returns) - 1)

        # Sharpe (not annualized for episode)
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1) if len(returns) > 1 else 1e-10
        episode_sharpe = mean_return / max(std_return, 1e-10)

        return {
            "episode_return": cumulative_return,
            "episode_sharpe": float(episode_sharpe),
            "avg_turnover": float(np.mean(turnovers)),
            "n_weeks": len(returns),
        }


def create_env_from_data(
    prices: dict[str, np.ndarray],
    signals: dict[str, dict[str, np.ndarray]],
    lstm_forecasts: dict[str, np.ndarray],
    patchtst_forecasts: dict[str, np.ndarray],
    symbol_order: list[str],
    config: PPOBaseConfig | None = None,
) -> PortfolioEnv:
    """Create environment from raw data dictionaries.

    Helper function to convert from dict format to array format.

    Args:
        prices: Dict of symbol -> array of prices (for computing returns).
        signals: Dict of symbol -> dict of signal_name -> array of values.
        lstm_forecasts: Dict of symbol -> array of LSTM forecast values.
        patchtst_forecasts: Dict of symbol -> array of PatchTST forecast values.
        symbol_order: Ordered list of symbols.
        config: PPO configuration.

    Returns:
        PortfolioEnv instance.
    """
    # Determine number of weeks from the first symbol
    first_symbol = symbol_order[0]
    n_weeks = len(prices[first_symbol]) - 1  # -1 because we compute returns
    n_stocks = len(symbol_order)
    n_signals = 7  # news + 5 fundamentals + fundamental_age

    # Build returns array
    symbol_returns = np.zeros((n_weeks, n_stocks))
    for stock_idx, symbol in enumerate(symbol_order):
        price_series = prices[symbol]
        # Weekly returns
        returns = (price_series[1:] - price_series[:-1]) / price_series[:-1]
        symbol_returns[:, stock_idx] = returns[:n_weeks]

    # Build signals array
    signal_names = [
        "news_sentiment",
        "gross_margin",
        "operating_margin",
        "net_margin",
        "current_ratio",
        "debt_to_equity",
        "fundamental_age",
    ]
    signals_array = np.zeros((n_weeks, n_stocks, n_signals))
    for stock_idx, symbol in enumerate(symbol_order):
        symbol_signals = signals.get(symbol, {})
        for signal_idx, signal_name in enumerate(signal_names):
            signal_values = symbol_signals.get(signal_name, np.zeros(n_weeks))
            signals_array[:, stock_idx, signal_idx] = signal_values[:n_weeks]

    # Build LSTM forecasts array
    lstm_array = np.zeros((n_weeks, n_stocks))
    for stock_idx, symbol in enumerate(symbol_order):
        forecast_values = lstm_forecasts.get(symbol, np.zeros(n_weeks))
        lstm_array[:, stock_idx] = forecast_values[:n_weeks]

    # Build PatchTST forecasts array
    patchtst_array = np.zeros((n_weeks, n_stocks))
    for stock_idx, symbol in enumerate(symbol_order):
        forecast_values = patchtst_forecasts.get(symbol, np.zeros(n_weeks))
        patchtst_array[:, stock_idx] = forecast_values[:n_weeks]

    return PortfolioEnv(
        symbol_returns=symbol_returns,
        signals=signals_array,
        lstm_forecasts=lstm_array,
        patchtst_forecasts=patchtst_array,
        symbol_order=symbol_order,
        config=config,
    )

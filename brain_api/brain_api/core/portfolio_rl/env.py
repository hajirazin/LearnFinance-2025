"""Weekly portfolio environment for RL training.

This environment simulates weekly portfolio rebalancing with:
- Long-only simplex weights + CASH
- Transaction costs
- Constraint enforcement (long-only simplex and cash buffer)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from brain_api.core.portfolio_rl.broker_costs import (
    IBKRSingaporeCostConfig,
    compute_ibkr_rebalance_cost,
)
from brain_api.core.portfolio_rl.config import DEFAULT_RL_BASE_CONFIG, RLBaseConfig
from brain_api.core.portfolio_rl.constraints import (
    apply_softmax_to_weights,
    compute_turnover,
    enforce_constraints,
)
from brain_api.core.portfolio_rl.rewards import (
    RebalanceTransition,
    compute_net_log_reward,
)
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
        patchtst_forecasts: np.ndarray,
        prices: np.ndarray,
        symbol_order: list[str],
        config: RLBaseConfig | None = None,
        cost_config: IBKRSingaporeCostConfig | None = None,
        max_episode_weeks: int | None = 52,
    ):
        """Initialize environment.

        Args:
            symbol_returns: Weekly returns for each symbol.
                           Shape: (n_weeks, n_stocks).
            signals: Per-stock signals for each week.
                    Shape: (n_weeks, n_stocks, n_signals_per_stock).
            patchtst_forecasts: PatchTST forecast feature for each stock each week.
                               Shape: (n_weeks, n_stocks).
            prices: Per-symbol close levels (NOT returns), shape
                ``(n_weeks, n_stocks)``. Required by the IBKR-SG cost
                model to convert weight deltas into share counts; per
                AGENTS.md rule #1 we will not silently default to a
                synthetic price grid.
            symbol_order: Ordered list of stock symbols.
            config: RL configuration.
            cost_config: IBKR Singapore Tiered cost schedule. Defaults
                to :meth:`IBKRSingaporeCostConfig.default` (calibrated
                to USD 10k NAV).
        """
        self.symbol_returns = symbol_returns
        self.signals = signals
        self.patchtst_forecasts = patchtst_forecasts
        self.prices = prices
        self.symbol_order = symbol_order
        self.config = config or DEFAULT_RL_BASE_CONFIG
        self.cost_config = cost_config or IBKRSingaporeCostConfig.default()
        self.max_episode_weeks = max_episode_weeks

        self.n_weeks = symbol_returns.shape[0]
        self.n_stocks = len(symbol_order)

        if prices.shape != (self.n_weeks, self.n_stocks):
            raise ValueError(
                f"prices shape {prices.shape} does not match "
                f"(n_weeks, n_stocks) = ({self.n_weeks}, {self.n_stocks})"
            )
        if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
            raise ValueError("prices must be complete, finite, and positive")

        self.schema = StateSchema(n_stocks=self.n_stocks)

        expected_signals = self.schema.n_signals_per_stock
        expected_shapes = {
            "symbol_returns": (self.n_weeks, self.n_stocks),
            "signals": (self.n_weeks, self.n_stocks, expected_signals),
            "patchtst_forecasts": (self.n_weeks, self.n_stocks),
        }
        for name, expected in expected_shapes.items():
            value = getattr(self, name)
            if value.shape != expected:
                raise ValueError(
                    f"{name} shape {value.shape} does not match {expected}"
                )
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be complete and finite")

        # Episode state
        self.current_week_idx: int = 0
        self.current_weights: np.ndarray = self._initial_weights()
        self.episode_start_week: int = 0

        # For tracking
        self.episode_returns: list[float] = []
        self.episode_turnovers: list[float] = []
        # Cache of last completed episode metrics (survives reset)
        self._last_episode_metrics: dict[str, float] | None = None

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

        # Get PatchTST forecast features for this week
        week_patchtst = self.patchtst_forecasts[week_idx]  # (n_stocks,)
        patchtst_dict = {
            symbol: float(week_patchtst[stock_idx])
            for stock_idx, symbol in enumerate(self.symbol_order)
        }

        return build_state_vector(
            signals=signals_dict,
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

        # Snapshot completed episode metrics before clearing
        if len(self.episode_returns) > 0:
            self._last_episode_metrics = self._compute_episode_metrics()

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
        )

        # Compute turnover (kept for episode statistics + info dict; no
        # longer drives the cost calculation -- IBKR-SG cost is per-leg).
        turnover = compute_turnover(self.current_weights, target_weights)

        # Get weekly returns for stocks (CASH return = 0)
        stock_returns = self.symbol_returns[self.current_week_idx]  # (n_stocks,)
        asset_returns = np.zeros(self.action_dim)
        asset_returns[: self.n_stocks] = stock_returns
        # CASH return is 0 (could add risk-free rate if desired)

        portfolio_return = float(np.dot(target_weights, asset_returns))

        # IBKR Singapore Tiered transaction cost (per-symbol, per-leg).
        # The cost model needs today's prices to convert weight deltas
        # into shares + per-order minimum charges. See
        # brain_api/core/portfolio_rl/broker_costs.py.
        rebalance_cost = compute_ibkr_rebalance_cost(
            symbol_order=self.symbol_order,
            current_weights=self.current_weights,
            target_weights=target_weights,
            prices=self.prices[self.current_week_idx],
            cfg=self.cost_config,
        )
        tc_fraction = rebalance_cost.total_fraction

        transition = RebalanceTransition.calculate(
            target_weights=target_weights,
            stock_returns=stock_returns,
            cost_fraction=tc_fraction,
        )
        reward = compute_net_log_reward(
            gross_return=transition.gross_return,
            transaction_cost_fraction=tc_fraction,
            config=self.config,
            target_weights=target_weights,
        )

        # Track for episode statistics
        net_portfolio_return = transition.net_growth - 1.0
        self.episode_returns.append(net_portfolio_return)
        self.episode_turnovers.append(turnover)

        # Update state
        self.current_weights = transition.post_weights
        self.current_week_idx += 1

        # Check if episode is done
        # Done if: (1) 52 weeks passed, or (2) end of data
        weeks_in_episode = self.current_week_idx - self.episode_start_week
        done = self.current_week_idx >= self.n_weeks
        if self.max_episode_weeks is not None:
            done = done or weeks_in_episode >= self.max_episode_weeks

        # Build next state (if not done)
        if done:
            next_state = np.zeros(self.state_dim)  # Dummy state
        else:
            next_state = self._build_state(self.current_week_idx)

        info = {
            "portfolio_return": portfolio_return,
            "net_portfolio_return": net_portfolio_return,
            "net_log_return": transition.net_log_return,
            "post_weights": transition.post_weights.tolist(),
            "turnover": turnover,
            "target_weights": target_weights.tolist(),
            "week_idx": self.current_week_idx - 1,
            "transaction_cost_usd": rebalance_cost.total_usd,
            "transaction_cost_fraction": tc_fraction,
            "cost_breakdown": rebalance_cost.breakdown(),
        }

        return EnvStep(
            next_state=next_state,
            reward=reward,
            done=done,
            info=info,
        )

    def _compute_episode_metrics(self) -> dict[str, float]:
        """Compute metrics from current episode_returns/episode_turnovers."""
        returns = np.array(self.episode_returns)
        turnovers = np.array(self.episode_turnovers)

        if len(returns) == 0:
            return {
                "episode_return": 0.0,
                "episode_sharpe": 0.0,
                "avg_turnover": 0.0,
                "n_weeks": 0,
            }

        cumulative_return = float(np.prod(1 + returns) - 1)
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1) if len(returns) > 1 else 1e-10
        episode_sharpe = mean_return / max(std_return, 1e-10)

        return {
            "episode_return": cumulative_return,
            "episode_sharpe": float(episode_sharpe),
            "avg_turnover": float(np.mean(turnovers)),
            "n_weeks": len(returns),
        }

    def get_episode_metrics(self) -> dict[str, float]:
        """Get metrics for the most recently completed episode.

        If the current episode has data, returns its metrics.
        If the current episode is empty (just after reset()), returns
        the cached metrics from the last completed episode.

        Returns:
            Dict with episode statistics.
        """
        if len(self.episode_returns) > 0:
            return self._compute_episode_metrics()

        # Current episode is empty (reset() just cleared it) --
        # return cached metrics from the last completed episode
        if self._last_episode_metrics is not None:
            return self._last_episode_metrics

        return {
            "episode_return": 0.0,
            "episode_sharpe": 0.0,
            "avg_turnover": 0.0,
            "n_weeks": 0,
        }


def create_env_from_data(
    prices: dict[str, np.ndarray],
    signals: dict[str, dict[str, np.ndarray]],
    patchtst_forecasts: dict[str, np.ndarray],
    symbol_order: list[str],
    config: RLBaseConfig | None = None,
    cost_config: IBKRSingaporeCostConfig | None = None,
) -> PortfolioEnv:
    """Create environment from raw data dictionaries.

    Helper function to convert from dict format to array format.

    Args:
        prices: Dict of symbol -> array of prices (for computing returns
            and for sizing trades against the IBKR-SG cost model).
        signals: Dict of symbol -> dict of signal_name -> array of values.
        patchtst_forecasts: Dict of symbol -> array of PatchTST forecast values.
        symbol_order: Ordered list of symbols.
        config: RL configuration.
        cost_config: IBKR Singapore Tiered cost schedule (defaults applied
            inside :class:`PortfolioEnv` if omitted).

    Returns:
        PortfolioEnv instance.
    """
    # Determine number of weeks from the first symbol
    first_symbol = symbol_order[0]
    n_weeks = len(prices[first_symbol]) - 1  # -1 because we compute returns
    n_stocks = len(symbol_order)
    schema = StateSchema(n_stocks=n_stocks)
    n_signals = schema.n_signals_per_stock

    # Build returns array + parallel prices array (close at the end of each
    # week, used by the IBKR-SG cost model to size shares per leg).
    symbol_returns = np.zeros((n_weeks, n_stocks))
    weekly_prices = np.zeros((n_weeks, n_stocks))
    for stock_idx, symbol in enumerate(symbol_order):
        price_series = prices[symbol]
        # Weekly returns
        returns = (price_series[1:] - price_series[:-1]) / price_series[:-1]
        symbol_returns[:, stock_idx] = returns[:n_weeks]
        # Use the start-of-week close (price at index t) so the price
        # available at the rebalance aligns with when the trade is executed
        # (at the start of week t).
        weekly_prices[:, stock_idx] = price_series[:n_weeks]

    # Build signals array
    signal_names = schema.signal_names
    signals_array = np.zeros((n_weeks, n_stocks, n_signals))
    for stock_idx, symbol in enumerate(symbol_order):
        if symbol not in signals:
            raise ValueError(f"Missing required SAC signals for {symbol}")
        symbol_signals = signals[symbol]
        for signal_idx, signal_name in enumerate(signal_names):
            if signal_name not in symbol_signals:
                raise ValueError(
                    f"Missing required SAC signal {signal_name!r} for {symbol}"
                )
            signal_values = symbol_signals[signal_name]
            if len(signal_values) < n_weeks:
                raise ValueError(
                    f"SAC signal {signal_name!r} for {symbol} has "
                    f"{len(signal_values)} values; expected at least {n_weeks}"
                )
            signals_array[:, stock_idx, signal_idx] = signal_values[:n_weeks]

    # Build PatchTST forecasts array
    patchtst_array = np.zeros((n_weeks, n_stocks))
    for stock_idx, symbol in enumerate(symbol_order):
        if (
            symbol not in patchtst_forecasts
            or len(patchtst_forecasts[symbol]) < n_weeks
        ):
            raise ValueError(f"Missing or short PatchTST forecasts for {symbol}")
        forecast_values = patchtst_forecasts[symbol]
        patchtst_array[:, stock_idx] = forecast_values[:n_weeks]

    return PortfolioEnv(
        symbol_returns=symbol_returns,
        signals=signals_array,
        patchtst_forecasts=patchtst_array,
        prices=weekly_prices,
        symbol_order=symbol_order,
        config=config,
        cost_config=cost_config,
    )

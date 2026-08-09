"""SAC training implementation with PatchTST-only forecasts.

Trains a SAC policy for portfolio allocation using PatchTST
predictions as the sole forecast features in the state vector.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from brain_api.core.portfolio_rl.env import PortfolioEnv
from brain_api.core.portfolio_rl.eval import (
    compute_cagr,
    compute_max_drawdown,
    compute_sharpe_ratio,
)
from brain_api.core.portfolio_rl.sac_networks import GaussianActor, TwinCritic
from brain_api.core.portfolio_rl.sac_trainer import SACTrainer
from brain_api.core.portfolio_rl.scaler import PortfolioScaler
from brain_api.core.portfolio_rl.state import StateSchema
from brain_api.core.sac.config import SACConfig
from brain_api.core.sac.regime_hmm import (
    RegimeHMMArtifact,
    causal_filter,
    fit_regime_hmm,
    market_observations,
    regime_probabilities,
)


@dataclass
class SACTrainingResult:
    """Result of SAC training with PatchTST-only forecasts."""

    actor: GaussianActor  # trained policy
    critic: TwinCritic  # trained critics
    critic_target: TwinCritic  # target critics
    log_alpha: torch.Tensor  # entropy coefficient
    scaler: PortfolioScaler  # fitted state scaler
    config: SACConfig
    symbol_order: list[str]  # ordered list of symbols
    regime_hmm: RegimeHMMArtifact
    audit_metadata: dict[str, Any]

    # Training metrics
    final_actor_loss: float
    final_critic_loss: float
    avg_episode_return: float
    avg_episode_sharpe: float

    # Evaluation metrics (on held-out data)
    eval_sharpe: float
    eval_cagr: float
    eval_max_drawdown: float


@dataclass
class TrainingData:
    """Prepared training data for SAC with PatchTST-only forecasts."""

    # Arrays aligned by week index
    symbol_returns: np.ndarray  # (n_weeks, n_stocks)
    signals: np.ndarray  # (n_weeks, n_stocks, n_signals)
    patchtst_forecasts: np.ndarray  # (n_weeks, n_stocks)
    # Rebalance-time Monday open prices (NOT returns); required by the IBKR-SG
    # cost model in PortfolioEnv.step to convert weight deltas into
    # share counts. Shape (n_weeks, n_stocks).
    prices: np.ndarray
    asset_masks: np.ndarray

    # Metadata
    symbol_order: list[str]
    n_weeks: int
    n_stocks: int
    weekly_dates: list[Any] | None = None
    market_dates: list[Any] | None = None
    spy_adjusted_closes: np.ndarray | None = None
    vix_closes: np.ndarray | None = None


def build_training_data(
    prices: dict[str, np.ndarray],
    signals: dict[str, dict[str, np.ndarray]],
    patchtst_predictions: dict[str, np.ndarray],
    symbol_order: list[str],
    *,
    weekly_dates: list[Any] | None = None,
    market_dates: list[Any] | None = None,
    spy_adjusted_closes: np.ndarray | None = None,
    vix_closes: np.ndarray | None = None,
) -> TrainingData:
    """Build training data arrays from raw data.

    Args:
        prices: Dict of symbol -> array of weekly prices.
        signals: Dict of symbol -> dict of signal_name -> array of signal values.
        patchtst_predictions: Dict of symbol -> array of PatchTST weekly return predictions.
        symbol_order: Ordered list of symbols to include.

    Returns:
        TrainingData with aligned arrays.
    """
    n_stocks = len(symbol_order)
    if n_stocks == 0:
        raise ValueError("symbol_order must be non-empty")
    if len(set(symbol_order)) != n_stocks:
        raise ValueError("symbol_order must contain unique symbols")
    if symbol_order != sorted(symbol_order):
        raise ValueError("SAC v3 symbol_order must be canonical lexicographic order")

    # Determine number of weeks from first symbol's prices
    first_symbol = symbol_order[0]
    n_weeks = len(prices[first_symbol]) - 1  # -1 because returns need two points
    if weekly_dates is not None and len(weekly_dates) != n_weeks:
        raise ValueError(f"weekly_dates must contain exactly {n_weeks} values")

    schema = StateSchema(n_stocks=n_stocks)
    signal_names = schema.signal_names
    n_signals = len(signal_names)

    # Build returns array + parallel rebalance-time price array (price[t],
    # lined up with the transition starting at t -- consumed
    # by the IBKR-SG cost model in PortfolioEnv.step).
    symbol_returns = np.zeros((n_weeks, n_stocks))
    weekly_prices = np.zeros((n_weeks, n_stocks))
    for stock_idx, symbol in enumerate(symbol_order):
        if symbol not in prices:
            raise ValueError(f"Missing required prices for {symbol}")
        price_series = np.asarray(prices[symbol], dtype=float)
        if len(price_series) != n_weeks + 1:
            raise ValueError(
                f"Prices for {symbol} have {len(price_series)} values; "
                f"expected {n_weeks + 1}"
            )
        if not np.all(np.isfinite(price_series)) or np.any(price_series <= 0):
            raise ValueError(f"Prices for {symbol} must be finite and positive")
        returns = (price_series[1:] - price_series[:-1]) / price_series[:-1]
        if not np.all(np.isfinite(returns)):
            raise ValueError(f"Returns for {symbol} must be complete and finite")
        symbol_returns[:, stock_idx] = returns
        weekly_prices[:, stock_idx] = price_series[:-1]

    # Build signals array
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
            signal_values = np.asarray(symbol_signals[signal_name], dtype=float)
            if len(signal_values) != n_weeks:
                raise ValueError(
                    f"SAC signal {signal_name!r} for {symbol} must contain "
                    f"exactly {n_weeks} values"
                )
            signals_array[:, stock_idx, signal_idx] = signal_values

    # Build PatchTST forecast features array
    patchtst_array = np.zeros((n_weeks, n_stocks))
    asset_masks = np.ones((n_weeks, n_stocks), dtype=bool)
    for stock_idx, symbol in enumerate(symbol_order):
        patchtst_preds = np.asarray(
            patchtst_predictions.get(symbol, np.full(n_weeks, np.nan)), dtype=float
        )
        if len(patchtst_preds) != n_weeks:
            raise ValueError(
                f"PatchTST forecasts for {symbol} must contain exactly {n_weeks} values"
            )
        finite = np.isfinite(patchtst_preds)
        asset_masks[:, stock_idx] &= finite
        patchtst_array[:, stock_idx] = np.where(finite, patchtst_preds, 0.0)

    # Missing/non-finite price-derived features make an asset ineligible for
    # that week. Provider-checked news keys are still required above.
    finite_signals = np.all(np.isfinite(signals_array), axis=2)
    asset_masks &= finite_signals
    signals_array = np.where(np.isfinite(signals_array), signals_array, 0.0)
    if np.any(asset_masks.sum(axis=1) < 10):
        bad = int(np.flatnonzero(asset_masks.sum(axis=1) < 10)[0])
        raise ValueError(
            f"SAC v3 training week {bad} has fewer than 10 eligible assets"
        )

    return TrainingData(
        symbol_returns=symbol_returns,
        signals=signals_array,
        patchtst_forecasts=patchtst_array,
        prices=weekly_prices,
        asset_masks=asset_masks,
        symbol_order=symbol_order,
        n_weeks=n_weeks,
        n_stocks=n_stocks,
        weekly_dates=(
            None
            if weekly_dates is None
            else [
                value.date() if hasattr(value, "date") else value
                for value in weekly_dates
            ]
        ),
        market_dates=(
            None
            if market_dates is None
            else [
                value.date() if hasattr(value, "date") else value
                for value in market_dates
            ]
        ),
        spy_adjusted_closes=spy_adjusted_closes,
        vix_closes=vix_closes,
    )


def create_env_from_training_data(
    training_data: TrainingData,
    config: SACConfig,
    start_week: int = 0,
    end_week: int | None = None,
) -> PortfolioEnv:
    """Create portfolio environment from training data.

    Args:
        training_data: Prepared training data.
        config: SAC configuration.
        start_week: Start week index.
        end_week: End week index (exclusive).

    Returns:
        Portfolio environment.
    """
    if end_week is None:
        end_week = training_data.n_weeks

    # Slice data for the specified window
    symbol_returns = training_data.symbol_returns[start_week:end_week]
    signals = training_data.signals[start_week:end_week]
    patchtst_forecasts = training_data.patchtst_forecasts[start_week:end_week]
    prices = training_data.prices[start_week:end_week]
    asset_masks = training_data.asset_masks[start_week:end_week]
    regime = getattr(training_data, "regime_probabilities", None)
    regime_slice = None if regime is None else regime[start_week:end_week]

    return PortfolioEnv(
        symbol_returns=symbol_returns,
        signals=signals,
        patchtst_forecasts=patchtst_forecasts,
        prices=prices,
        symbol_order=training_data.symbol_order,
        config=config,
        asset_masks=asset_masks,
        regime_probabilities=regime_slice,
    )


def train_sac(
    training_data: TrainingData,
    config: SACConfig,
    shutdown_event: threading.Event | None = None,
) -> SACTrainingResult:
    """Train SAC model with PatchTST-only forecasts.

    Args:
        training_data: Prepared training data with PatchTST forecasts.
        config: SAC configuration.

    Returns:
        Training result with trained models.
    """
    # Seed before environment construction. SACTrainer seeds again before
    # network construction; this also keeps environment initialization stable.
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Locked policy: deterministic chronological trailing 104-week net OOS eval.
    eval_weeks = 104
    train_weeks = training_data.n_weeks - eval_weeks
    if train_weeks < 104:
        raise ValueError(
            "SAC training requires at least 208 weekly transitions: "
            "104 train and 104 deterministic OOS evaluation"
        )

    print(
        f"[SAC] Training on {train_weeks} weeks, evaluating on {training_data.n_weeks - train_weeks} weeks"
    )

    if (
        training_data.weekly_dates is None
        or training_data.market_dates is None
        or training_data.spy_adjusted_closes is None
        or training_data.vix_closes is None
    ):
        raise ValueError("SAC v3 training requires aligned SPY/VIX market history")
    market_dates = list(training_data.market_dates)
    spy = np.asarray(training_data.spy_adjusted_closes, dtype=float)
    vix = np.asarray(training_data.vix_closes, dtype=float)
    if len(market_dates) != len(spy) or len(spy) != len(vix):
        raise ValueError("SAC v3 market dates/SPY/VIX arrays must align")
    observations = market_observations(spy, vix)
    observation_dates = market_dates[20:]
    cutoff = training_data.weekly_dates[train_weeks - 1]
    train_observation_count = sum(value <= cutoff for value in observation_dates)
    if train_observation_count < 3:
        raise ValueError("insufficient train-fold HMM observations")
    tail_indices = [
        index for index, value in enumerate(market_dates) if value <= cutoff
    ][-21:]
    if len(tail_indices) != 21:
        raise ValueError("SAC v3 HMM requires 21 market-tail sessions at cutoff")
    regime_hmm = fit_regime_hmm(
        observations[:train_observation_count],
        observation_dates[:train_observation_count],
        spy_tail=spy[tail_indices],
        vix_tail=vix[tail_indices],
        tail_dates=[market_dates[index] for index in tail_indices],
    )
    train_posteriors = causal_filter(observations[:train_observation_count], regime_hmm)
    if train_observation_count < len(observations):
        live_posteriors = causal_filter(
            observations[train_observation_count:],
            regime_hmm,
            regime_hmm.terminal_posterior,
        )
        all_posteriors = np.vstack((train_posteriors, live_posteriors))
    else:
        all_posteriors = train_posteriors
    weekly_regime = np.empty((training_data.n_weeks, 2), dtype=float)
    for index, weekly_date in enumerate(training_data.weekly_dates):
        posterior_index = (
            np.searchsorted(observation_dates, weekly_date, side="right") - 1
        )
        if posterior_index < 0:
            raise ValueError(f"no causal HMM posterior available for {weekly_date}")
        weekly_regime[index] = regime_probabilities(
            all_posteriors[posterior_index], regime_hmm
        )
    training_data.regime_probabilities = weekly_regime

    # Create training environment
    train_env = create_env_from_training_data(
        training_data,
        config,
        start_week=0,
        end_week=train_weeks,
    )

    # Fit on each training-fold week exactly once. The PatchTST median is
    # independent of portfolio actions, so sampling environment states would
    # both duplicate weeks and make the statistic policy-path dependent.
    scaler = PortfolioScaler.create(n_stocks=training_data.n_stocks)
    training_medians = np.asarray(
        [
            np.median(
                training_data.patchtst_forecasts[index][
                    training_data.asset_masks[index]
                ]
            )
            for index in range(train_weeks)
        ],
        dtype=np.float64,
    )
    scaler.fit_patchtst_medians(training_medians)

    # Create normalized environment wrapper
    train_env_normalized = NormalizedEnv(train_env, scaler)

    # Train SAC
    trainer = SACTrainer(train_env_normalized, config, shutdown_event=shutdown_event)
    trainer.train(total_timesteps=config.total_timesteps)

    # Get trained models
    sac_result = trainer.get_result()

    # Evaluate on held-out data
    eval_env = create_env_from_training_data(
        training_data,
        config,
        start_week=train_weeks,
        end_week=training_data.n_weeks,
    )
    eval_env.max_episode_weeks = None
    eval_env_normalized = NormalizedEnv(eval_env, scaler)

    eval_sharpe, eval_cagr, eval_max_drawdown = evaluate_policy(
        sac_result.actor,
        eval_env_normalized,
        config,
        expected_periods=eval_weeks,
    )

    print(
        f"[SAC] Eval sharpe: {eval_sharpe:.4f}, CAGR: {eval_cagr * 100:.2f}%, Max DD: {eval_max_drawdown * 100:.2f}%"
    )

    return SACTrainingResult(
        actor=sac_result.actor,
        critic=sac_result.critic,
        critic_target=sac_result.critic_target,
        log_alpha=sac_result.log_alpha,
        scaler=scaler,
        config=config,
        symbol_order=training_data.symbol_order,
        regime_hmm=regime_hmm,
        audit_metadata={
            "sac_schema_version": 3,
            "architecture": "masked_attention",
            "training_cutoff_date": cutoff.isoformat(),
            "canonical_symbol_order": list(training_data.symbol_order),
        },
        final_actor_loss=sac_result.final_actor_loss,
        final_critic_loss=sac_result.final_critic_loss,
        avg_episode_return=sac_result.avg_episode_return,
        avg_episode_sharpe=sac_result.avg_episode_sharpe,
        eval_sharpe=eval_sharpe,
        eval_cagr=eval_cagr,
        eval_max_drawdown=eval_max_drawdown,
    )


def evaluate_policy(
    actor: GaussianActor,
    env: NormalizedEnv,
    config: SACConfig,
    expected_periods: int | None = None,
) -> tuple[float, float, float]:
    """Evaluate policy on environment.

    Args:
        actor: Trained actor network.
        env: Normalized environment.
        config: Configuration.

    Returns:
        Tuple of (sharpe, cagr, max_drawdown).
    """
    portfolio_returns = []

    state = env.reset(start_week=0)
    done = False

    while not done:
        # Get deterministic action
        action = actor.get_action(state, deterministic=True)

        # Step environment
        step_result = env.step(action)
        state = step_result.next_state
        done = step_result.done

        # Evaluation and promotion use exact after-cost returns.
        portfolio_returns.append(step_result.info["net_portfolio_return"])

    portfolio_returns = np.array(portfolio_returns)
    if expected_periods is not None and len(portfolio_returns) != expected_periods:
        raise ValueError(
            f"Deterministic SAC evaluation produced {len(portfolio_returns)} "
            f"periods; expected {expected_periods}"
        )

    # Compute metrics
    sharpe = compute_sharpe_ratio(portfolio_returns)
    cagr = compute_cagr(portfolio_returns)
    max_dd = compute_max_drawdown(portfolio_returns)

    return sharpe, cagr, max_dd


@dataclass
class StepResult:
    """Result from environment step."""

    next_state: np.ndarray
    reward: float
    done: bool
    portfolio_return: float
    info: dict[str, Any]


class NormalizedEnv:
    """Wrapper that normalizes states using a fitted scaler."""

    def __init__(self, env: PortfolioEnv, scaler: PortfolioScaler):
        """Initialize normalized environment.

        Args:
            env: Base environment.
            scaler: Fitted state scaler.
        """
        self.env = env
        self.scaler = scaler

    @property
    def state_dim(self) -> int:
        return self.env.state_dim

    @property
    def action_dim(self) -> int:
        return self.env.action_dim

    def reset(self, start_week: int | None = None) -> np.ndarray:
        """Reset and return normalized state."""
        state = self.env.reset(start_week=start_week)
        return self.scaler.transform(state)

    def step(self, action: np.ndarray) -> StepResult:
        """Step and return normalized next state."""
        result = self.env.step(action)
        # Create new result with normalized state
        # portfolio_return is in the info dict
        return StepResult(
            next_state=self.scaler.transform(result.next_state),
            reward=result.reward,
            done=result.done,
            portfolio_return=result.info.get("portfolio_return", 0.0),
            info=result.info,
        )

    def get_episode_metrics(self) -> dict[str, float]:
        """Get episode metrics from underlying env."""
        return self.env.get_episode_metrics()

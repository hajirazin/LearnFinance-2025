"""SAC training implementation with dual forecasts (LSTM + PatchTST).

Trains a SAC policy for portfolio allocation using both LSTM and PatchTST
predictions as forecast features in the state vector.
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
from brain_api.core.portfolio_rl.sac_config import SACFinetuneConfig
from brain_api.core.portfolio_rl.sac_networks import GaussianActor, TwinCritic
from brain_api.core.portfolio_rl.sac_trainer import SACTrainer
from brain_api.core.portfolio_rl.scaler import PortfolioScaler
from brain_api.core.portfolio_rl.state import StateSchema
from brain_api.core.sac.config import SACConfig


@dataclass
class SACTrainingResult:
    """Result of SAC training with dual forecasts."""

    actor: GaussianActor  # trained policy
    critic: TwinCritic  # trained critics
    critic_target: TwinCritic  # target critics
    log_alpha: torch.Tensor  # entropy coefficient
    scaler: PortfolioScaler  # fitted state scaler
    config: SACConfig
    symbol_order: list[str]  # ordered list of symbols

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
    """Prepared training data for SAC with dual forecasts."""

    # Arrays aligned by week index
    symbol_returns: np.ndarray  # (n_weeks, n_stocks)
    signals: np.ndarray  # (n_weeks, n_stocks, n_signals)
    lstm_forecasts: np.ndarray  # (n_weeks, n_stocks)
    patchtst_forecasts: np.ndarray  # (n_weeks, n_stocks)
    # Rebalance-time Monday open prices (NOT returns); required by the IBKR-SG
    # cost model in PortfolioEnv.step to convert weight deltas into
    # share counts. Shape (n_weeks, n_stocks).
    prices: np.ndarray

    # Metadata
    symbol_order: list[str]
    n_weeks: int
    n_stocks: int


def build_training_data(
    prices: dict[str, np.ndarray],
    signals: dict[str, dict[str, np.ndarray]],
    lstm_predictions: dict[str, np.ndarray],
    patchtst_predictions: dict[str, np.ndarray],
    symbol_order: list[str],
) -> TrainingData:
    """Build training data arrays from raw data.

    Args:
        prices: Dict of symbol -> array of weekly prices.
        signals: Dict of symbol -> dict of signal_name -> array of signal values.
        lstm_predictions: Dict of symbol -> array of LSTM weekly return predictions.
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

    # Determine number of weeks from first symbol's prices
    first_symbol = symbol_order[0]
    n_weeks = len(prices[first_symbol]) - 1  # -1 because returns need two points

    schema = StateSchema(n_stocks=n_stocks, schema_version=2)
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
            if len(signal_values) != n_weeks or not np.all(np.isfinite(signal_values)):
                raise ValueError(
                    f"SAC signal {signal_name!r} for {symbol} must contain "
                    f"exactly {n_weeks} finite values"
                )
            signals_array[:, stock_idx, signal_idx] = signal_values

    # Build LSTM forecast features array
    lstm_array = np.zeros((n_weeks, n_stocks))
    for stock_idx, symbol in enumerate(symbol_order):
        if symbol not in lstm_predictions:
            raise ValueError(f"Missing required LSTM forecasts for {symbol}")
        lstm_preds = np.asarray(lstm_predictions[symbol], dtype=float)
        if len(lstm_preds) != n_weeks or not np.all(np.isfinite(lstm_preds)):
            raise ValueError(
                f"LSTM forecasts for {symbol} must contain exactly "
                f"{n_weeks} finite values"
            )
        lstm_array[:, stock_idx] = lstm_preds

    # Build PatchTST forecast features array
    patchtst_array = np.zeros((n_weeks, n_stocks))
    for stock_idx, symbol in enumerate(symbol_order):
        if symbol not in patchtst_predictions:
            raise ValueError(f"Missing required PatchTST forecasts for {symbol}")
        patchtst_preds = np.asarray(patchtst_predictions[symbol], dtype=float)
        if len(patchtst_preds) != n_weeks or not np.all(np.isfinite(patchtst_preds)):
            raise ValueError(
                f"PatchTST forecasts for {symbol} must contain exactly "
                f"{n_weeks} finite values"
            )
        patchtst_array[:, stock_idx] = patchtst_preds

    return TrainingData(
        symbol_returns=symbol_returns,
        signals=signals_array,
        lstm_forecasts=lstm_array,
        patchtst_forecasts=patchtst_array,
        prices=weekly_prices,
        symbol_order=symbol_order,
        n_weeks=n_weeks,
        n_stocks=n_stocks,
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
    lstm_forecasts = training_data.lstm_forecasts[start_week:end_week]
    patchtst_forecasts = training_data.patchtst_forecasts[start_week:end_week]
    prices = training_data.prices[start_week:end_week]

    return PortfolioEnv(
        symbol_returns=symbol_returns,
        signals=signals,
        lstm_forecasts=lstm_forecasts,
        patchtst_forecasts=patchtst_forecasts,
        prices=prices,
        symbol_order=training_data.symbol_order,
        config=config,
        schema_version=2,
    )


def train_sac(
    training_data: TrainingData,
    config: SACConfig,
    shutdown_event: threading.Event | None = None,
) -> SACTrainingResult:
    """Train SAC model with dual forecasts.

    Args:
        training_data: Prepared training data with dual forecasts.
        config: SAC configuration.

    Returns:
        Training result with trained models.
    """
    # Seed before environment reset/scaler sampling. SACTrainer seeds again
    # before network construction, but waiting until then makes each candidate's
    # scaler depend on process-global RNG state and preceding candidates.
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

    # Create training environment
    train_env = create_env_from_training_data(
        training_data,
        config,
        start_week=0,
        end_week=train_weeks,
    )

    # Create and fit scaler on training data
    scaler = PortfolioScaler.create(n_stocks=training_data.n_stocks, schema_version=2)
    # Collect sample states for fitting
    sample_states = []
    state = train_env.reset()
    sample_states.append(state)
    for _ in range(min(100, train_weeks)):
        action = np.random.randn(train_env.action_dim)
        step_result = train_env.step(action)
        sample_states.append(step_result.next_state)
        if step_result.done:
            state = train_env.reset()
    scaler.fit(np.array(sample_states))

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
        final_actor_loss=sac_result.final_actor_loss,
        final_critic_loss=sac_result.final_critic_loss,
        avg_episode_return=sac_result.avg_episode_return,
        avg_episode_sharpe=sac_result.avg_episode_sharpe,
        eval_sharpe=eval_sharpe,
        eval_cagr=eval_cagr,
        eval_max_drawdown=eval_max_drawdown,
    )


def finetune_sac(
    training_data: TrainingData,
    prior_actor: GaussianActor,
    prior_critic: TwinCritic,
    prior_critic_target: TwinCritic,
    prior_log_alpha: torch.Tensor,
    prior_scaler: PortfolioScaler,
    prior_config: SACConfig,
    finetune_config: SACFinetuneConfig,
    shutdown_event: threading.Event | None = None,
) -> SACTrainingResult:
    """Fine-tune SAC model on recent data.

    Args:
        training_data: Recent training data with dual forecasts.
        prior_actor: Previously trained actor.
        prior_critic: Previously trained critics.
        prior_critic_target: Previously trained target critics.
        prior_log_alpha: Previously trained entropy coefficient.
        prior_scaler: Previously fitted scaler.
        prior_config: Previous configuration.
        finetune_config: Fine-tuning configuration.

    Returns:
        Fine-tuned model result.
    """
    # Create environment from recent data
    env = create_env_from_training_data(training_data, prior_config)
    env_normalized = NormalizedEnv(env, prior_scaler)

    # Create trainer with prior models
    trainer = SACTrainer(env_normalized, prior_config, shutdown_event=shutdown_event)

    # Load prior weights
    trainer.actor.load_state_dict(prior_actor.state_dict())
    trainer.critic.load_state_dict(prior_critic.state_dict())
    trainer.critic_target.load_state_dict(prior_critic_target.state_dict())
    trainer.log_alpha = prior_log_alpha.clone().requires_grad_(True)

    # Update learning rates for fine-tuning
    for param_group in trainer.actor_optimizer.param_groups:
        param_group["lr"] = finetune_config.actor_lr
    for param_group in trainer.critic_optimizer.param_groups:
        param_group["lr"] = finetune_config.critic_lr
    if trainer.alpha_optimizer is not None:
        for param_group in trainer.alpha_optimizer.param_groups:
            param_group["lr"] = finetune_config.alpha_lr

    # Fine-tune
    trainer.train(total_timesteps=finetune_config.total_timesteps)

    # Get result
    sac_result = trainer.get_result()

    # Evaluate
    eval_sharpe, eval_cagr, eval_max_drawdown = evaluate_policy(
        sac_result.actor,
        env_normalized,
        prior_config,
    )

    return SACTrainingResult(
        actor=sac_result.actor,
        critic=sac_result.critic,
        critic_target=sac_result.critic_target,
        log_alpha=sac_result.log_alpha,
        scaler=prior_scaler,  # Keep same scaler
        config=prior_config,
        symbol_order=training_data.symbol_order,
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

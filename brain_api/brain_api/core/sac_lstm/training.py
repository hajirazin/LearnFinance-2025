"""SAC + LSTM training implementation.

Trains a SAC policy for portfolio allocation using LSTM predictions
as a forecast feature in the state vector.
"""

from __future__ import annotations

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
from brain_api.core.sac_lstm.config import SACLSTMConfig


@dataclass
class SACLSTMTrainingResult:
    """Result of SAC + LSTM training."""

    actor: GaussianActor  # trained policy
    critic: TwinCritic  # trained critics
    critic_target: TwinCritic  # target critics
    log_alpha: torch.Tensor  # entropy coefficient
    scaler: PortfolioScaler  # fitted state scaler
    config: SACLSTMConfig
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
    """Prepared training data for SAC."""

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
        signals: Dict of symbol -> dict of signal_name -> array of signal values.
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
            returns = (price_series[1:] - price_series[:-1]) / np.maximum(
                price_series[:-1], 1e-10
            )
            symbol_returns[: len(returns), stock_idx] = returns[:n_weeks]

    # Build signals array
    signals_array = np.zeros((n_weeks, n_stocks, n_signals))
    for stock_idx, symbol in enumerate(symbol_order):
        symbol_signals = signals.get(symbol, {})
        for signal_idx, signal_name in enumerate(signal_names):
            signal_values = symbol_signals.get(signal_name)
            if signal_values is not None:
                # Assume signals are weekly-aligned
                signals_array[: len(signal_values), stock_idx, signal_idx] = (
                    signal_values[:n_weeks]
                )

    # Build forecast features array
    forecast_array = np.zeros((n_weeks, n_stocks))
    for stock_idx, symbol in enumerate(symbol_order):
        lstm_preds = lstm_predictions.get(symbol)
        if lstm_preds is not None:
            forecast_array[: len(lstm_preds), stock_idx] = lstm_preds[:n_weeks]

    return TrainingData(
        symbol_returns=symbol_returns,
        signals=signals_array,
        forecast_features=forecast_array,
        symbol_order=symbol_order,
        n_weeks=n_weeks,
        n_stocks=n_stocks,
    )


def create_env_from_training_data(
    training_data: TrainingData,
    config: SACLSTMConfig,
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
    # PortfolioEnv expects numpy arrays, not dicts
    symbol_returns = training_data.symbol_returns[start_week:end_week]
    signals = training_data.signals[start_week:end_week]
    forecast_features = training_data.forecast_features[start_week:end_week]

    return PortfolioEnv(
        symbol_returns=symbol_returns,
        signals=signals,
        forecast_features=forecast_features,
        symbol_order=training_data.symbol_order,
        config=config,
    )


def train_sac_lstm(
    training_data: TrainingData,
    config: SACLSTMConfig,
) -> SACLSTMTrainingResult:
    """Train SAC + LSTM model.

    Args:
        training_data: Prepared training data.
        config: SAC + LSTM configuration.

    Returns:
        Training result with trained models.
    """
    # Split data: train on first portion, evaluate on last validation_years
    weeks_per_year = 52
    eval_weeks = config.validation_years * weeks_per_year
    train_weeks = max(training_data.n_weeks - eval_weeks, weeks_per_year * 2)

    print(
        f"[SAC_LSTM] Training on {train_weeks} weeks, evaluating on {training_data.n_weeks - train_weeks} weeks"
    )

    # Create training environment
    train_env = create_env_from_training_data(
        training_data,
        config,
        start_week=0,
        end_week=train_weeks,
    )

    # Create and fit scaler on training data
    scaler = PortfolioScaler()
    # Collect sample states for fitting
    sample_states = []
    state = train_env.reset()
    sample_states.append(state)
    for _ in range(min(100, train_weeks)):
        action = np.random.randn(config.action_dim)
        step_result = train_env.step(action)
        sample_states.append(step_result.next_state)
        if step_result.done:
            state = train_env.reset()
    scaler.fit(np.array(sample_states))

    # Create normalized environment wrapper
    train_env_normalized = NormalizedEnv(train_env, scaler)

    # Train SAC
    trainer = SACTrainer(train_env_normalized, config)
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
    eval_env_normalized = NormalizedEnv(eval_env, scaler)

    eval_sharpe, eval_cagr, eval_max_drawdown = evaluate_policy(
        sac_result.actor,
        eval_env_normalized,
        config,
    )

    print(
        f"[SAC_LSTM] Eval sharpe: {eval_sharpe:.4f}, CAGR: {eval_cagr * 100:.2f}%, Max DD: {eval_max_drawdown * 100:.2f}%"
    )

    return SACLSTMTrainingResult(
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


def finetune_sac_lstm(
    training_data: TrainingData,
    prior_actor: GaussianActor,
    prior_critic: TwinCritic,
    prior_critic_target: TwinCritic,
    prior_log_alpha: torch.Tensor,
    prior_scaler: PortfolioScaler,
    prior_config: SACLSTMConfig,
    finetune_config: SACFinetuneConfig,
) -> SACLSTMTrainingResult:
    """Fine-tune SAC + LSTM model on recent data.

    Args:
        training_data: Recent training data.
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
    trainer = SACTrainer(env_normalized, prior_config)

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

    return SACLSTMTrainingResult(
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
    config: SACLSTMConfig,
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

    state = env.reset()
    done = False

    while not done:
        # Get deterministic action
        action = actor.get_action(state, deterministic=True)

        # Step environment
        step_result = env.step(action)
        state = step_result.next_state
        done = step_result.done

        # Collect portfolio return for this week
        portfolio_returns.append(step_result.portfolio_return)

    portfolio_returns = np.array(portfolio_returns)

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

    def reset(self) -> np.ndarray:
        """Reset and return normalized state."""
        state = self.env.reset()
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

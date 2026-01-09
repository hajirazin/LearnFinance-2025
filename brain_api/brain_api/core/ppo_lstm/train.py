"""PPO + LSTM full training implementation.

Contains the main train_ppo_lstm function and related utilities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from brain_api.core.portfolio_rl.env import PortfolioEnv
from brain_api.core.portfolio_rl.scaler import PortfolioScaler
from brain_api.core.ppo_lstm.config import PPOLSTMConfig
from brain_api.core.ppo_lstm.data import TrainingData
from brain_api.core.ppo_lstm.model import PPOActorCritic
from brain_api.core.ppo_lstm.trainer import PPOTrainer


@dataclass
class PPOTrainingResult:
    """Result of PPO training."""

    model: PPOActorCritic  # trained policy
    scaler: PortfolioScaler  # fitted state scaler
    config: PPOLSTMConfig
    symbol_order: list[str]  # ordered list of symbols

    # Training metrics
    final_policy_loss: float
    final_value_loss: float
    avg_episode_return: float
    avg_episode_sharpe: float

    # Evaluation metrics (on held-out data)
    eval_sharpe: float
    eval_cagr: float
    eval_max_drawdown: float


def evaluate_policy(
    model: PPOActorCritic,
    env: PortfolioEnv,
    scaler: PortfolioScaler,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate policy on environment.

    Args:
        model: Trained PPO model.
        env: Environment to evaluate on.
        scaler: State scaler.
        device: PyTorch device.

    Returns:
        Dict with evaluation metrics.
    """
    model.eval()
    model.to(device)

    state = env.reset(start_week=0)  # Start from beginning

    returns = []

    done = False
    while not done:
        # Scale state
        scaled_state = scaler.transform(state)
        state_t = torch.FloatTensor(scaled_state).to(device)

        # Get deterministic action
        with torch.no_grad():
            action, _, _ = model.get_action_and_value(state_t, deterministic=True)

        action_np = action.cpu().numpy().flatten()

        # Step
        step_result = env.step(action_np)
        returns.append(step_result.info["portfolio_return"])

        if step_result.done:
            done = True
        else:
            state = step_result.next_state

    returns = np.array(returns)

    # Compute metrics
    from brain_api.core.portfolio_rl.eval import (
        compute_cagr,
        compute_max_drawdown,
        compute_sharpe_ratio,
    )

    return {
        "sharpe": compute_sharpe_ratio(returns),
        "cagr": compute_cagr(returns),
        "max_drawdown": compute_max_drawdown(returns),
    }


def train_ppo_lstm(
    training_data: TrainingData,
    config: PPOLSTMConfig,
    validation_split: float = 0.2,
) -> PPOTrainingResult:
    """Train PPO + LSTM portfolio allocator.

    Args:
        training_data: Prepared training data.
        config: PPO configuration.
        validation_split: Fraction of data to use for validation.

    Returns:
        PPOTrainingResult with trained model and metrics.
    """
    # Split data (time-based, not random)
    n_weeks = training_data.n_weeks
    split_idx = int(n_weeks * (1 - validation_split))

    train_returns = training_data.symbol_returns[:split_idx]
    train_signals = training_data.signals[:split_idx]
    train_forecasts = training_data.forecast_features[:split_idx]

    val_returns = training_data.symbol_returns[split_idx:]
    val_signals = training_data.signals[split_idx:]
    val_forecasts = training_data.forecast_features[split_idx:]

    print(f"[PPO] Data split: {split_idx} train weeks, {n_weeks - split_idx} val weeks")

    # Create training environment
    train_env = PortfolioEnv(
        symbol_returns=train_returns,
        signals=train_signals,
        forecast_features=train_forecasts,
        symbol_order=training_data.symbol_order,
        config=config,
    )

    # Create scaler and fit on training states
    scaler = PortfolioScaler.create(n_stocks=training_data.n_stocks)

    # Collect sample states to fit scaler
    sample_states = []
    state = train_env.reset()
    for _ in range(min(100, split_idx)):
        sample_states.append(state)
        # Random action for sampling
        action = np.random.randn(train_env.action_dim)
        step_result = train_env.step(action)
        state = train_env.reset() if step_result.done else step_result.next_state

    scaler.fit(np.array(sample_states))

    # Train PPO
    trainer = PPOTrainer(train_env, config)
    history = trainer.train()

    # Move model to CPU for storage
    model_cpu = PPOActorCritic(
        state_dim=train_env.state_dim,
        action_dim=train_env.action_dim,
        hidden_sizes=config.hidden_sizes,
        activation=config.activation,
    )
    model_cpu.load_state_dict({
        k: v.cpu() for k, v in trainer.model.state_dict().items()
    })

    # Evaluate on validation data
    val_env = PortfolioEnv(
        symbol_returns=val_returns,
        signals=val_signals,
        forecast_features=val_forecasts,
        symbol_order=training_data.symbol_order,
        config=config,
    )

    eval_metrics = evaluate_policy(model_cpu, val_env, scaler, trainer.device)

    # Compute final metrics
    final_policy_loss = history["policy_loss"][-1] if history["policy_loss"] else float("inf")
    final_value_loss = history["value_loss"][-1] if history["value_loss"] else float("inf")
    avg_episode_return = np.mean(history["episode_return"]) if history["episode_return"] else 0.0
    avg_episode_sharpe = np.mean(history["episode_sharpe"]) if history["episode_sharpe"] else 0.0

    print(f"[PPO] Final metrics: policy_loss={final_policy_loss:.4f}, value_loss={final_value_loss:.4f}")
    print(f"[PPO] Eval: sharpe={eval_metrics['sharpe']:.4f}, cagr={eval_metrics['cagr']*100:.2f}%, max_dd={eval_metrics['max_drawdown']*100:.2f}%")

    return PPOTrainingResult(
        model=model_cpu,
        scaler=scaler,
        config=config,
        symbol_order=training_data.symbol_order,
        final_policy_loss=final_policy_loss,
        final_value_loss=final_value_loss,
        avg_episode_return=avg_episode_return,
        avg_episode_sharpe=avg_episode_sharpe,
        eval_sharpe=eval_metrics["sharpe"],
        eval_cagr=eval_metrics["cagr"],
        eval_max_drawdown=eval_metrics["max_drawdown"],
    )


"""PPO + LSTM fine-tuning implementation.

Provides weekly fine-tuning capability for the PPO policy,
loading a prior model and continuing training on recent data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from brain_api.core.portfolio_rl.config import PPOConfig
from brain_api.core.portfolio_rl.env import PortfolioEnv
from brain_api.core.portfolio_rl.scaler import PortfolioScaler
from brain_api.core.ppo_lstm.config import PPOLSTMConfig
from brain_api.core.ppo_lstm.data import TrainingData
from brain_api.core.ppo_lstm.model import PPOActorCritic
from brain_api.core.ppo_lstm.train import PPOTrainingResult, evaluate_policy
from brain_api.core.ppo_lstm.trainer import PPOTrainer
from brain_api.core.training_utils import get_device


@dataclass
class PPOFinetuneConfig:
    """Configuration specifically for fine-tuning.
    
    Uses shorter lookback and fewer training steps than full training.
    """
    
    # Fine-tune on last 26 weeks (6 months rolling buffer)
    lookback_weeks: int = 26
    
    # Fewer timesteps since we're continuing from a trained model
    total_timesteps: int = 2_000
    
    # Smaller learning rate for fine-tuning
    learning_rate: float = 1e-4
    
    # Other parameters inherited from prior model's config


def finetune_ppo_lstm(
    training_data: TrainingData,
    prior_model: PPOActorCritic,
    prior_scaler: PortfolioScaler,
    prior_config: PPOLSTMConfig,
    finetune_config: PPOFinetuneConfig | None = None,
) -> PPOTrainingResult:
    """Fine-tune PPO + LSTM on recent data.
    
    Continues training from a prior model using a rolling 26-week buffer.
    This is called weekly (Sunday cron) to adapt to recent market conditions.
    
    Args:
        training_data: Recent training data (26 weeks).
        prior_model: Previously trained PPO model to continue from.
        prior_scaler: Scaler from prior model.
        prior_config: Config from prior model.
        finetune_config: Fine-tuning specific settings.
    
    Returns:
        PPOTrainingResult with fine-tuned model and metrics.
    """
    if finetune_config is None:
        finetune_config = PPOFinetuneConfig()
    
    # Use recent data only (last 26 weeks)
    n_weeks = min(training_data.n_weeks, finetune_config.lookback_weeks)
    
    # Split: use last 4 weeks for validation
    val_weeks = 4
    split_idx = n_weeks - val_weeks
    
    train_returns = training_data.symbol_returns[-n_weeks:-val_weeks]
    train_signals = training_data.signals[-n_weeks:-val_weeks]
    train_forecasts = training_data.forecast_features[-n_weeks:-val_weeks]
    
    val_returns = training_data.symbol_returns[-val_weeks:]
    val_signals = training_data.signals[-val_weeks:]
    val_forecasts = training_data.forecast_features[-val_weeks:]
    
    print(f"[PPO Finetune] Using {split_idx} train weeks, {val_weeks} val weeks")
    
    # Create training environment
    train_env = PortfolioEnv(
        symbol_returns=train_returns,
        signals=train_signals,
        forecast_features=train_forecasts,
        symbol_order=training_data.symbol_order,
        config=prior_config,
    )
    
    # Reuse prior scaler (already fitted on full history)
    scaler = prior_scaler
    
    # Create trainer with prior model weights
    device = get_device()
    
    # Clone prior model and move to device
    model = PPOActorCritic(
        state_dim=train_env.state_dim,
        action_dim=train_env.action_dim,
        hidden_sizes=prior_config.hidden_sizes,
        activation=prior_config.activation,
    )
    model.load_state_dict(prior_model.state_dict())
    model.to(device)
    
    # Create trainer with lower learning rate
    finetune_ppo_config = PPOConfig(
        hidden_sizes=prior_config.hidden_sizes,
        activation=prior_config.activation,
        learning_rate=finetune_config.learning_rate,
        total_timesteps=finetune_config.total_timesteps,
        # Keep other params from prior
        clip_epsilon=prior_config.clip_epsilon,
        gae_lambda=prior_config.gae_lambda,
        gamma=prior_config.gamma,
        entropy_coef=prior_config.entropy_coef,
        value_coef=prior_config.value_coef,
        max_grad_norm=prior_config.max_grad_norm,
        n_epochs=prior_config.n_epochs,
        batch_size=prior_config.batch_size,
        rollout_steps=prior_config.rollout_steps,
        cost_bps=prior_config.cost_bps,
        cash_buffer=prior_config.cash_buffer,
        max_position_weight=prior_config.max_position_weight,
        reward_scale=prior_config.reward_scale,
        n_stocks=prior_config.n_stocks,
        seed=prior_config.seed,
    )
    
    trainer = PPOTrainer(train_env, finetune_ppo_config, device=device)
    # Replace trainer's model with our pre-loaded one
    trainer.model = model
    trainer.optimizer = torch.optim.Adam(
        trainer.model.parameters(),
        lr=finetune_config.learning_rate,
    )
    
    # Fine-tune
    history = trainer.train(total_timesteps=finetune_config.total_timesteps)
    
    # Move model to CPU for storage
    model_cpu = PPOActorCritic(
        state_dim=train_env.state_dim,
        action_dim=train_env.action_dim,
        hidden_sizes=prior_config.hidden_sizes,
        activation=prior_config.activation,
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
        config=prior_config,
    )
    
    eval_metrics = evaluate_policy(model_cpu, val_env, scaler, trainer.device)
    
    # Compute final metrics
    final_policy_loss = history["policy_loss"][-1] if history["policy_loss"] else float("inf")
    final_value_loss = history["value_loss"][-1] if history["value_loss"] else float("inf")
    avg_episode_return = np.mean(history["episode_return"]) if history["episode_return"] else 0.0
    avg_episode_sharpe = np.mean(history["episode_sharpe"]) if history["episode_sharpe"] else 0.0
    
    print(f"[PPO Finetune] Final metrics: policy_loss={final_policy_loss:.4f}, value_loss={final_value_loss:.4f}")
    print(f"[PPO Finetune] Eval: sharpe={eval_metrics['sharpe']:.4f}, cagr={eval_metrics['cagr']*100:.2f}%, max_dd={eval_metrics['max_drawdown']*100:.2f}%")
    
    return PPOTrainingResult(
        model=model_cpu,
        scaler=scaler,
        config=prior_config,
        symbol_order=training_data.symbol_order,
        final_policy_loss=final_policy_loss,
        final_value_loss=final_value_loss,
        avg_episode_return=avg_episode_return,
        avg_episode_sharpe=avg_episode_sharpe,
        eval_sharpe=eval_metrics["sharpe"],
        eval_cagr=eval_metrics["cagr"],
        eval_max_drawdown=eval_metrics["max_drawdown"],
    )


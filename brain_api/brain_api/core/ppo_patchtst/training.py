"""PPO + PatchTST training implementation.

Trains a PPO policy for portfolio allocation using PatchTST predictions
as a forecast feature in the state vector.

This is structurally identical to ppo_lstm training, but uses
PatchTST forecasts instead of LSTM forecasts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from brain_api.core.portfolio_rl.env import PortfolioEnv
from brain_api.core.portfolio_rl.scaler import PortfolioScaler
from brain_api.core.ppo_patchtst.config import PPOPatchTSTConfig
from brain_api.core.ppo_lstm.model import PPOActorCritic
from brain_api.core.ppo_lstm.data import TrainingData, build_training_data
from brain_api.core.ppo_lstm.trainer import PPOTrainer
from brain_api.core.ppo_lstm.finetune import PPOFinetuneConfig
from brain_api.core.ppo_lstm.train import evaluate_policy
from brain_api.core.portfolio_rl.config import PPOConfig
from brain_api.core.training_utils import get_device
import torch


@dataclass
class PPOPatchTSTTrainingResult:
    """Result of PPO + PatchTST training."""
    
    model: PPOActorCritic  # trained policy
    scaler: PortfolioScaler  # fitted state scaler
    config: PPOPatchTSTConfig
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


def train_ppo_patchtst(
    training_data: TrainingData,
    config: PPOPatchTSTConfig,
    validation_split: float = 0.2,
) -> PPOPatchTSTTrainingResult:
    """Train PPO + PatchTST portfolio allocator.
    
    This is functionally identical to train_ppo_lstm, but:
    - Uses PatchTST forecasts as the forecast feature
    - Returns PPOPatchTSTTrainingResult
    
    Args:
        training_data: Prepared training data with PatchTST forecasts.
        config: PPO configuration.
        validation_split: Fraction of data to use for validation.
    
    Returns:
        PPOPatchTSTTrainingResult with trained model and metrics.
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
    
    print(f"[PPO_PatchTST] Data split: {split_idx} train weeks, {n_weeks - split_idx} val weeks")
    
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
        action = np.random.randn(train_env.action_dim)
        step_result = train_env.step(action)
        if step_result.done:
            state = train_env.reset()
        else:
            state = step_result.next_state
    
    scaler.fit(np.array(sample_states))
    
    # Train PPO (reuse trainer from ppo_lstm)
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
    
    print(f"[PPO_PatchTST] Final metrics: policy_loss={final_policy_loss:.4f}, value_loss={final_value_loss:.4f}")
    print(f"[PPO_PatchTST] Eval: sharpe={eval_metrics['sharpe']:.4f}, cagr={eval_metrics['cagr']*100:.2f}%, max_dd={eval_metrics['max_drawdown']*100:.2f}%")
    
    return PPOPatchTSTTrainingResult(
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


# ============================================================================
# Fine-tuning (weekly, loads prior model)
# ============================================================================


def finetune_ppo_patchtst(
    training_data: TrainingData,
    prior_model: PPOActorCritic,
    prior_scaler: PortfolioScaler,
    prior_config: PPOPatchTSTConfig,
    finetune_config: PPOFinetuneConfig | None = None,
) -> PPOPatchTSTTrainingResult:
    """Fine-tune PPO + PatchTST on recent data.
    
    Continues training from a prior model using a rolling 26-week buffer.
    This is called weekly (Sunday cron) to adapt to recent market conditions.
    
    Args:
        training_data: Recent training data (26 weeks).
        prior_model: Previously trained PPO model to continue from.
        prior_scaler: Scaler from prior model.
        prior_config: Config from prior model.
        finetune_config: Fine-tuning specific settings.
    
    Returns:
        PPOPatchTSTTrainingResult with fine-tuned model and metrics.
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
    
    print(f"[PPO_PatchTST Finetune] Using {split_idx} train weeks, {val_weeks} val weeks")
    
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
    
    print(f"[PPO_PatchTST Finetune] Final: policy_loss={final_policy_loss:.4f}, value_loss={final_value_loss:.4f}")
    print(f"[PPO_PatchTST Finetune] Eval: sharpe={eval_metrics['sharpe']:.4f}, cagr={eval_metrics['cagr']*100:.2f}%, max_dd={eval_metrics['max_drawdown']*100:.2f}%")
    
    return PPOPatchTSTTrainingResult(
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


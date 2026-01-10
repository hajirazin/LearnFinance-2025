"""SAC + PatchTST training implementation.

Trains a SAC policy for portfolio allocation using PatchTST predictions
as a forecast feature in the state vector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from brain_api.core.portfolio_rl.sac_config import SACFinetuneConfig
from brain_api.core.portfolio_rl.sac_networks import GaussianActor, TwinCritic
from brain_api.core.portfolio_rl.sac_trainer import SACTrainer
from brain_api.core.portfolio_rl.scaler import PortfolioScaler

# Reuse training infrastructure from sac_lstm
from brain_api.core.sac_lstm.training import (
    NormalizedEnv,
    TrainingData,
    create_env_from_training_data,
    evaluate_policy,
)
from brain_api.core.sac_patchtst.config import SACPatchTSTConfig


@dataclass
class SACPatchTSTTrainingResult:
    """Result of SAC + PatchTST training."""

    actor: GaussianActor  # trained policy
    critic: TwinCritic  # trained critics
    critic_target: TwinCritic  # target critics
    log_alpha: torch.Tensor  # entropy coefficient
    scaler: PortfolioScaler  # fitted state scaler
    config: SACPatchTSTConfig
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


def train_sac_patchtst(
    training_data: TrainingData,
    config: SACPatchTSTConfig,
) -> SACPatchTSTTrainingResult:
    """Train SAC + PatchTST model.

    Args:
        training_data: Prepared training data.
        config: SAC + PatchTST configuration.

    Returns:
        Training result with trained models.
    """
    # Split data: train on first portion, evaluate on last validation_years
    weeks_per_year = 52
    eval_weeks = config.validation_years * weeks_per_year
    train_weeks = max(training_data.n_weeks - eval_weeks, weeks_per_year * 2)

    print(
        f"[SAC_PatchTST] Training on {train_weeks} weeks, evaluating on {training_data.n_weeks - train_weeks} weeks"
    )

    # Create training environment
    train_env = create_env_from_training_data(
        training_data,
        config,
        start_week=0,
        end_week=train_weeks,
    )

    # Create and fit scaler on training data
    scaler = PortfolioScaler.create(n_stocks=training_data.n_stocks)
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
        f"[SAC_PatchTST] Eval sharpe: {eval_sharpe:.4f}, CAGR: {eval_cagr * 100:.2f}%, Max DD: {eval_max_drawdown * 100:.2f}%"
    )

    return SACPatchTSTTrainingResult(
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


def finetune_sac_patchtst(
    training_data: TrainingData,
    prior_actor: GaussianActor,
    prior_critic: TwinCritic,
    prior_critic_target: TwinCritic,
    prior_log_alpha: torch.Tensor,
    prior_scaler: PortfolioScaler,
    prior_config: SACPatchTSTConfig,
    finetune_config: SACFinetuneConfig,
) -> SACPatchTSTTrainingResult:
    """Fine-tune SAC + PatchTST model on recent data.

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

    return SACPatchTSTTrainingResult(
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

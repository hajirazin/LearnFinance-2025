"""PPO + LSTM training implementation.

Trains a PPO policy for portfolio allocation using LSTM predictions
as a forecast feature in the state vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from brain_api.core.portfolio_rl.config import PPOConfig
from brain_api.core.portfolio_rl.env import PortfolioEnv
from brain_api.core.portfolio_rl.scaler import PortfolioScaler
from brain_api.core.portfolio_rl.state import StateSchema
from brain_api.core.ppo_lstm.config import PPOLSTMConfig
from brain_api.core.ppo_lstm.model import PPOActorCritic
from brain_api.core.training_utils import get_device


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


@dataclass
class TrainingData:
    """Prepared training data for PPO."""
    
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
        signals: Dict of symbol -> dict of signal_name -> array of daily values.
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
            returns = (price_series[1:] - price_series[:-1]) / np.maximum(price_series[:-1], 1e-10)
            symbol_returns[:len(returns), stock_idx] = returns[:n_weeks]
    
    # Build signals array
    signals_array = np.zeros((n_weeks, n_stocks, n_signals))
    for stock_idx, symbol in enumerate(symbol_order):
        symbol_signals = signals.get(symbol, {})
        for signal_idx, signal_name in enumerate(signal_names):
            signal_values = symbol_signals.get(signal_name)
            if signal_values is not None:
                # Assume signals are weekly-aligned
                signals_array[:len(signal_values), stock_idx, signal_idx] = signal_values[:n_weeks]
    
    # Build forecast features array
    forecast_array = np.zeros((n_weeks, n_stocks))
    for stock_idx, symbol in enumerate(symbol_order):
        lstm_preds = lstm_predictions.get(symbol)
        if lstm_preds is not None:
            forecast_array[:len(lstm_preds), stock_idx] = lstm_preds[:n_weeks]
    
    return TrainingData(
        symbol_returns=symbol_returns,
        signals=signals_array,
        forecast_features=forecast_array,
        symbol_order=symbol_order,
        n_weeks=n_weeks,
        n_stocks=n_stocks,
    )


class PPOTrainer:
    """PPO trainer for portfolio allocation."""
    
    def __init__(
        self,
        env: PortfolioEnv,
        config: PPOLSTMConfig,
        device: torch.device | None = None,
    ):
        """Initialize trainer.
        
        Args:
            env: Portfolio environment.
            config: PPO configuration.
            device: PyTorch device (auto-detected if None).
        """
        self.env = env
        self.config = config
        self.device = device or get_device()
        
        # Create actor-critic model
        self.model = PPOActorCritic(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            hidden_sizes=config.hidden_sizes,
            activation=config.activation,
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
        )
    
    def collect_rollout(
        self,
        n_steps: int,
    ) -> dict[str, np.ndarray]:
        """Collect rollout data from environment.
        
        Args:
            n_steps: Number of steps to collect.
        
        Returns:
            Dict with states, actions, rewards, values, log_probs, dones.
        """
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []
        
        state = self.env.reset()
        
        for _ in range(n_steps):
            # Convert state to tensor
            state_t = torch.FloatTensor(state).to(self.device)
            
            # Get action from policy
            with torch.no_grad():
                action, log_prob, value = self.model.get_action_and_value(state_t)
            
            action_np = action.cpu().numpy().flatten()
            
            # Step environment
            step_result = self.env.step(action_np)
            
            # Store data
            states.append(state)
            actions.append(action_np)
            rewards.append(step_result.reward)
            values.append(value.cpu().numpy().item())
            log_probs.append(log_prob.cpu().numpy().item())
            dones.append(step_result.done)
            
            # Update state
            if step_result.done:
                state = self.env.reset()
            else:
                state = step_result.next_state
        
        return {
            "states": np.array(states),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "values": np.array(values),
            "log_probs": np.array(log_probs),
            "dones": np.array(dones),
        }
    
    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_value: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation.
        
        Args:
            rewards: Array of rewards.
            values: Array of value estimates.
            dones: Array of done flags.
            last_value: Value estimate for final state.
        
        Returns:
            Tuple of (advantages, returns).
        """
        n_steps = len(rewards)
        advantages = np.zeros(n_steps)
        returns = np.zeros(n_steps)
        
        # Compute GAE backwards
        gae = 0
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(dones[t])
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - float(dones[t])
            
            delta = rewards[t] + self.config.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns
    
    def update(
        self,
        rollout: dict[str, np.ndarray],
    ) -> dict[str, float]:
        """Perform PPO update on rollout data.
        
        Args:
            rollout: Rollout data from collect_rollout.
        
        Returns:
            Dict with loss metrics.
        """
        # Convert to tensors
        states = torch.FloatTensor(rollout["states"]).to(self.device)
        actions = torch.FloatTensor(rollout["actions"]).to(self.device)
        old_log_probs = torch.FloatTensor(rollout["log_probs"]).to(self.device)
        
        # Compute advantages
        with torch.no_grad():
            last_state = torch.FloatTensor(rollout["states"][-1]).to(self.device)
            last_value = self.model.value_net(last_state).cpu().numpy().item()
        
        advantages, returns = self.compute_gae(
            rollout["rewards"],
            rollout["values"],
            rollout["dones"],
            last_value,
        )
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update epochs
        n_samples = len(states)
        indices = np.arange(n_samples)
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0
        
        for _ in range(self.config.n_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, n_samples, self.config.batch_size):
                end = min(start + self.config.batch_size, n_samples)
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate current policy
                log_probs, entropy, values = self.model.evaluate(
                    batch_states, batch_actions
                )
                
                # Policy loss (clipped surrogate objective)
                ratio = (log_probs - batch_old_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon,
                ) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = self.config.value_coef * nn.functional.mse_loss(
                    values, batch_returns
                )
                
                # Entropy bonus
                entropy_loss = -self.config.entropy_coef * entropy.mean()
                
                # Total loss
                loss = policy_loss + value_loss + entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1
        
        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }
    
    def train(
        self,
        total_timesteps: int | None = None,
    ) -> dict[str, list[float]]:
        """Train PPO for specified number of timesteps.
        
        Args:
            total_timesteps: Total timesteps to train (uses config if None).
        
        Returns:
            Dict with training history.
        """
        total_timesteps = total_timesteps or self.config.total_timesteps
        rollout_steps = self.config.rollout_steps
        
        n_rollouts = total_timesteps // rollout_steps
        
        history = {
            "policy_loss": [],
            "value_loss": [],
            "episode_return": [],
            "episode_sharpe": [],
        }
        
        print(f"[PPO] Starting training for {total_timesteps} timesteps ({n_rollouts} rollouts)")
        print(f"[PPO] Device: {self.device}")
        
        for rollout_idx in range(n_rollouts):
            # Collect rollout
            rollout = self.collect_rollout(rollout_steps)
            
            # Get episode metrics
            episode_metrics = self.env.get_episode_metrics()
            
            # Update policy
            update_metrics = self.update(rollout)
            
            # Record history
            history["policy_loss"].append(update_metrics["policy_loss"])
            history["value_loss"].append(update_metrics["value_loss"])
            history["episode_return"].append(episode_metrics["episode_return"])
            history["episode_sharpe"].append(episode_metrics["episode_sharpe"])
            
            # Log progress
            if (rollout_idx + 1) % max(1, n_rollouts // 10) == 0:
                print(
                    f"[PPO] Rollout {rollout_idx + 1}/{n_rollouts}: "
                    f"policy_loss={update_metrics['policy_loss']:.4f}, "
                    f"value_loss={update_metrics['value_loss']:.4f}, "
                    f"ep_return={episode_metrics['episode_return']:.4f}, "
                    f"ep_sharpe={episode_metrics['episode_sharpe']:.4f}"
                )
        
        print("[PPO] Training complete")
        return history


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
        if step_result.done:
            state = train_env.reset()
        else:
            state = step_result.next_state
    
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
        compute_sharpe_ratio,
        compute_cagr,
        compute_max_drawdown,
    )
    
    return {
        "sharpe": compute_sharpe_ratio(returns),
        "cagr": compute_cagr(returns),
        "max_drawdown": compute_max_drawdown(returns),
    }


# ============================================================================
# Fine-tuning (weekly, loads prior model)
# ============================================================================


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


"""SAC training loop for portfolio RL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from brain_api.core.portfolio_rl.constraints import (
    apply_softmax_to_weights,
    enforce_constraints,
)
from brain_api.core.portfolio_rl.sac_buffer import BatchSample, ReplayBuffer
from brain_api.core.portfolio_rl.sac_networks import (
    GaussianActor,
    TwinCritic,
    hard_update,
    soft_update,
)

if TYPE_CHECKING:
    from brain_api.core.portfolio_rl.env import PortfolioEnv
    from brain_api.core.portfolio_rl.sac_config import SACConfig


@dataclass
class SACTrainingResult:
    """Result from SAC training."""

    actor: GaussianActor
    critic: TwinCritic
    critic_target: TwinCritic
    log_alpha: torch.Tensor
    final_actor_loss: float
    final_critic_loss: float
    avg_episode_return: float
    avg_episode_sharpe: float


class SACTrainer:
    """Soft Actor-Critic trainer for portfolio allocation.

    Implements:
    - Twin Q-critics with target networks
    - Automatic entropy temperature tuning
    - Polyak averaging for target updates
    """

    def __init__(
        self,
        env: PortfolioEnv,
        config: SACConfig,
    ):
        """Initialize SAC trainer.

        Args:
            env: Portfolio environment.
            config: SAC configuration.
        """
        self.env = env
        self.config = config

        # Dimensions
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim

        # Set random seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # Initialize networks
        self.actor = GaussianActor(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_sizes=config.hidden_sizes,
            activation=config.activation,
        )

        self.critic = TwinCritic(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_sizes=config.hidden_sizes,
            activation=config.activation,
        )

        self.critic_target = TwinCritic(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_sizes=config.hidden_sizes,
            activation=config.activation,
        )

        # Initialize target with same weights
        hard_update(self.critic_target, self.critic)

        # Optimizers
        self.actor_optimizer = optim.AdamW(
            self.actor.parameters(),
            lr=config.actor_lr,
            weight_decay=config.weight_decay,
        )

        self.critic_optimizer = optim.AdamW(
            self.critic.parameters(),
            lr=config.critic_lr,
            weight_decay=config.weight_decay,
        )

        # Entropy coefficient (alpha)
        if config.auto_entropy_tuning:
            # Target entropy: -dim(action) is common default
            self.target_entropy = (
                config.target_entropy
                if config.target_entropy is not None
                else -float(self.action_dim)
            )
            # Log alpha as learnable parameter
            self.log_alpha = torch.tensor(
                np.log(config.init_alpha),
                dtype=torch.float32,
                requires_grad=True,
            )
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.alpha_lr)
        else:
            self.target_entropy = None
            self.log_alpha = torch.tensor(np.log(config.init_alpha), dtype=torch.float32)
            self.alpha_optimizer = None

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=config.buffer_size,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            seed=config.seed,
        )

        # Training stats
        self.total_steps = 0
        self.episode_returns: list[float] = []
        self.episode_sharpes: list[float] = []

    @property
    def alpha(self) -> float:
        """Current entropy coefficient."""
        return self.log_alpha.exp().item()

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Select action given state.

        Args:
            state: Current state.
            deterministic: If True, use mean action.

        Returns:
            Action (portfolio weight logits).
        """
        return self.actor.get_action(state, deterministic=deterministic)

    def _update(self, batch: BatchSample) -> tuple[float, float, float]:
        """Perform one SAC update step.

        Args:
            batch: Batch of transitions.

        Returns:
            Tuple of (critic_loss, actor_loss, alpha_loss).
        """
        # Convert to tensors
        states = torch.FloatTensor(batch.states)
        actions = torch.FloatTensor(batch.actions)
        rewards = torch.FloatTensor(batch.rewards).unsqueeze(-1)
        next_states = torch.FloatTensor(batch.next_states)
        dones = torch.FloatTensor(batch.dones).unsqueeze(-1)

        alpha = self.log_alpha.exp().detach()

        # === Update Critics ===
        with torch.no_grad():
            # Sample next actions from policy
            next_actions, next_log_probs = self.actor(next_states)

            # Compute target Q-values
            q1_target, q2_target = self.critic_target(next_states, next_actions)
            min_q_target = torch.min(q1_target, q2_target)

            # Soft target: Q - alpha * log_pi
            target_q = rewards + self.config.gamma * (1 - dones) * (
                min_q_target - alpha * next_log_probs.unsqueeze(-1)
            )

        # Current Q-values
        q1, q2 = self.critic(states, actions)

        # Critic loss
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # === Update Actor ===
        # Freeze critic during actor update
        for param in self.critic.parameters():
            param.requires_grad = False

        new_actions, log_probs = self.actor(states)
        q1_new, q2_new = self.critic(states, new_actions)
        min_q_new = torch.min(q1_new, q2_new)

        # Actor loss: maximize Q - alpha * log_pi
        actor_loss = (alpha * log_probs.unsqueeze(-1) - min_q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critic
        for param in self.critic.parameters():
            param.requires_grad = True

        # === Update Alpha (entropy coefficient) ===
        alpha_loss = 0.0
        if self.config.auto_entropy_tuning and self.alpha_optimizer is not None:
            # Alpha loss: minimize alpha * (log_pi + target_entropy)
            alpha_loss_tensor = -(
                self.log_alpha * (log_probs + self.target_entropy).detach()
            ).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss_tensor.backward()
            self.alpha_optimizer.step()

            alpha_loss = alpha_loss_tensor.item()

        # === Soft update target networks ===
        soft_update(self.critic_target, self.critic, self.config.tau)

        return critic_loss.item(), actor_loss.item(), alpha_loss

    def train(
        self,
        total_timesteps: int | None = None,
    ) -> dict[str, list[float]]:
        """Train SAC agent.

        Args:
            total_timesteps: Total environment steps for training.

        Returns:
            Training history dict.
        """
        if total_timesteps is None:
            total_timesteps = self.config.total_timesteps

        history = {
            "critic_loss": [],
            "actor_loss": [],
            "alpha": [],
            "episode_return": [],
        }

        state = self.env.reset()
        episode_return = 0.0

        for _step in range(total_timesteps):
            self.total_steps += 1

            # Select action
            if self.total_steps < self.config.warmup_steps:
                # Random action during warmup
                action = np.random.randn(self.action_dim)
            else:
                action = self.select_action(state)

            # Step environment
            step_result = self.env.step(action)
            next_state = step_result.next_state
            reward = step_result.reward
            done = step_result.done

            # Store transition
            self.replay_buffer.add(state, action, reward, next_state, done)

            episode_return += reward

            # Update if buffer is ready
            if self.replay_buffer.is_ready(self.config.batch_size):
                for _ in range(self.config.gradient_steps_per_env_step):
                    batch = self.replay_buffer.sample(self.config.batch_size)
                    critic_loss, actor_loss, _alpha_loss = self._update(batch)

                    history["critic_loss"].append(critic_loss)
                    history["actor_loss"].append(actor_loss)
                    history["alpha"].append(self.alpha)

            if done:
                # Episode finished
                episode_metrics = self.env.get_episode_metrics()
                self.episode_returns.append(episode_metrics["episode_return"])
                self.episode_sharpes.append(episode_metrics["episode_sharpe"])
                history["episode_return"].append(episode_metrics["episode_return"])

                # Reset for new episode
                state = self.env.reset()
                episode_return = 0.0
            else:
                state = next_state

        return history

    def get_result(self) -> SACTrainingResult:
        """Get training result with trained models."""
        return SACTrainingResult(
            actor=self.actor,
            critic=self.critic,
            critic_target=self.critic_target,
            log_alpha=self.log_alpha.detach().clone(),
            final_actor_loss=(
                sum(self.episode_returns[-10:]) / 10 if self.episode_returns else 0.0
            ),
            final_critic_loss=0.0,  # Not tracked per-episode
            avg_episode_return=(
                np.mean(self.episode_returns) if self.episode_returns else 0.0
            ),
            avg_episode_sharpe=(
                np.mean(self.episode_sharpes) if self.episode_sharpes else 0.0
            ),
        )


def action_to_weights(
    action: np.ndarray,
    cash_buffer: float = 0.02,
    max_position_weight: float = 0.20,
) -> np.ndarray:
    """Convert raw action logits to constrained portfolio weights.

    Args:
        action: Raw action logits from policy.
        cash_buffer: Minimum cash weight.
        max_position_weight: Maximum weight per position.

    Returns:
        Constrained portfolio weights that sum to 1.
    """
    # Apply softmax to get raw weights
    weights = apply_softmax_to_weights(action)

    # Apply constraints
    weights = enforce_constraints(
        weights,
        cash_buffer=cash_buffer,
        max_position_weight=max_position_weight,
    )

    return weights


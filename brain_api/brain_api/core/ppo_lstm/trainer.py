"""PPO Trainer class for portfolio allocation.

Contains the core PPO training algorithm with rollout collection,
GAE computation, and policy updates.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from brain_api.core.portfolio_rl.env import PortfolioEnv
from brain_api.core.ppo_lstm.config import PPOLSTMConfig
from brain_api.core.ppo_lstm.model import PPOActorCritic
from brain_api.core.training_utils import get_device


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
            state = self.env.reset() if step_result.done else step_result.next_state

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

            delta = (
                rewards[t]
                + self.config.gamma * next_value * next_non_terminal
                - values[t]
            )
            gae = (
                delta
                + self.config.gamma * self.config.gae_lambda * next_non_terminal * gae
            )
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
                surr2 = (
                    torch.clamp(
                        ratio,
                        1.0 - self.config.clip_epsilon,
                        1.0 + self.config.clip_epsilon,
                    )
                    * batch_advantages
                )
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

        print(
            f"[PPO] Starting training for {total_timesteps} timesteps ({n_rollouts} rollouts)"
        )
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

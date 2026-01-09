"""PPO neural network models.

Actor-Critic architecture for portfolio allocation:
- Actor (Policy): outputs portfolio weight logits
- Critic (Value): estimates state value for advantage computation
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


class PPOPolicy(nn.Module):
    """PPO policy (actor) network for portfolio allocation.

    Outputs mean and log_std for a Gaussian policy over action logits.
    Actions are then passed through softmax to get portfolio weights.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...] = (64, 64),
        activation: str = "tanh",
    ):
        """Initialize policy network.

        Args:
            state_dim: Dimension of state vector.
            action_dim: Dimension of action vector (n_stocks + 1 for CASH).
            hidden_sizes: Hidden layer sizes.
            activation: Activation function ("tanh" or "relu").
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Build MLP layers
        layers = []
        prev_dim = state_dim

        act_fn = nn.Tanh() if activation == "tanh" else nn.ReLU()

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_size))
            layers.append(act_fn)
            prev_dim = hidden_size

        self.net = nn.Sequential(*layers)

        # Output heads for mean and log_std
        self.mean_head = nn.Linear(prev_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            state: State tensor, shape (batch, state_dim) or (state_dim,).

        Returns:
            Tuple of (action_mean, action_log_std).
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        features = self.net(state)
        mean = self.mean_head(features)
        log_std = self.log_std.expand_as(mean)

        return mean, log_std

    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy.

        Args:
            state: State tensor.
            deterministic: If True, return mean action (no sampling).

        Returns:
            Tuple of (action, log_prob).
        """
        mean, log_std = self.forward(state)

        if deterministic:
            action = mean
            log_prob = torch.zeros(mean.shape[0], device=mean.device)
        else:
            std = log_std.exp()
            dist = Normal(mean, std)
            action = dist.rsample()  # reparameterized sample
            log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob

    def evaluate_actions(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log probability and entropy of actions.

        Args:
            state: State tensor.
            action: Action tensor.

        Returns:
            Tuple of (log_prob, entropy).
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy


class PPOValueNetwork(nn.Module):
    """PPO value (critic) network.

    Estimates state value V(s) for computing advantages.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_sizes: tuple[int, ...] = (64, 64),
        activation: str = "tanh",
    ):
        """Initialize value network.

        Args:
            state_dim: Dimension of state vector.
            hidden_sizes: Hidden layer sizes.
            activation: Activation function.
        """
        super().__init__()

        layers = []
        prev_dim = state_dim

        act_fn = nn.Tanh() if activation == "tanh" else nn.ReLU()

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_size))
            layers.append(act_fn)
            prev_dim = hidden_size

        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            state: State tensor.

        Returns:
            Value estimate, shape (batch, 1).
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.net(state)


class PPOActorCritic(nn.Module):
    """Combined Actor-Critic module for PPO.

    Combines policy and value networks with shared convenience methods.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...] = (64, 64),
        activation: str = "tanh",
    ):
        """Initialize actor-critic.

        Args:
            state_dim: Dimension of state vector.
            action_dim: Dimension of action vector.
            hidden_sizes: Hidden layer sizes (shared for actor and critic).
            activation: Activation function.
        """
        super().__init__()

        self.policy = PPOPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
        )

        self.value_net = PPOValueNetwork(
            state_dim=state_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
        )

    def forward(
        self,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for both policy and value.

        Args:
            state: State tensor.

        Returns:
            Tuple of (action_mean, action_log_std, value).
        """
        mean, log_std = self.policy(state)
        value = self.value_net(state)
        return mean, log_std, value

    def get_action_and_value(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action and compute value.

        Args:
            state: State tensor.
            deterministic: If True, return mean action.

        Returns:
            Tuple of (action, log_prob, value).
        """
        action, log_prob = self.policy.get_action(state, deterministic)
        value = self.value_net(state)
        return action, log_prob, value.squeeze(-1)

    def evaluate(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update.

        Args:
            state: State tensor.
            action: Action tensor.

        Returns:
            Tuple of (log_prob, entropy, value).
        """
        log_prob, entropy = self.policy.evaluate_actions(state, action)
        value = self.value_net(state)
        return log_prob, entropy, value.squeeze(-1)

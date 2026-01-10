"""Neural network architectures for SAC.

Implements:
- GaussianActor: Stochastic policy that outputs mean and log_std
- TwinCritic: Two Q-networks for double Q-learning
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

LOG_STD_MIN = -20
LOG_STD_MAX = 2


def create_mlp(
    input_dim: int,
    output_dim: int,
    hidden_sizes: tuple[int, ...],
    activation: str = "relu",
) -> nn.Sequential:
    """Create a multi-layer perceptron.

    Args:
        input_dim: Input dimension.
        output_dim: Output dimension.
        hidden_sizes: Tuple of hidden layer sizes.
        activation: Activation function ("relu" or "tanh").

    Returns:
        Sequential MLP module.
    """
    act_fn = nn.ReLU if activation == "relu" else nn.Tanh

    layers: list[nn.Module] = []
    prev_dim = input_dim

    for hidden_dim in hidden_sizes:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(act_fn())
        prev_dim = hidden_dim

    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    """Gaussian policy for SAC.

    Outputs a diagonal Gaussian distribution over actions (logits).
    Actions are sampled and then passed through softmax externally
    to produce portfolio weights.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...] = (64, 64),
        activation: str = "relu",
    ):
        """Initialize actor network.

        Args:
            state_dim: Dimension of state input.
            action_dim: Dimension of action output.
            hidden_sizes: Hidden layer sizes.
            activation: Activation function.
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared feature extraction
        act_fn = nn.ReLU if activation == "relu" else nn.Tanh

        layers: list[nn.Module] = []
        prev_dim = state_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_fn())
            prev_dim = hidden_dim

        self.feature_net = nn.Sequential(*layers)

        # Separate heads for mean and log_std
        self.mean_head = nn.Linear(prev_dim, action_dim)
        self.log_std_head = nn.Linear(prev_dim, action_dim)

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: sample action and compute log probability.

        Args:
            state: State tensor (batch_size, state_dim).
            deterministic: If True, return mean action without sampling.

        Returns:
            Tuple of (action, log_prob).
            - action: Sampled or mean action (batch_size, action_dim).
            - log_prob: Log probability of the action (batch_size,).
        """
        features = self.feature_net(state)

        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        if deterministic:
            action = mean
            # For deterministic, log_prob is not meaningful but we return 0
            log_prob = torch.zeros(state.shape[0], device=state.device)
        else:
            # Sample from Gaussian
            dist = Normal(mean, std)
            action = dist.rsample()  # Reparameterization trick

            # Compute log probability
            # Sum over action dimensions
            log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob

    def get_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Get action from state (numpy interface).

        Args:
            state: State array (state_dim,) or (batch_size, state_dim).
            deterministic: If True, return mean action.

        Returns:
            Action array.
        """
        with torch.no_grad():
            # Get device from model parameters
            device = next(self.parameters()).device
            state_tensor = torch.FloatTensor(state).to(device)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)

            action, _ = self.forward(state_tensor, deterministic=deterministic)
            return action.squeeze(0).cpu().numpy()


class Critic(nn.Module):
    """Q-network for SAC.

    Takes state and action as input, outputs Q-value.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...] = (64, 64),
        activation: str = "relu",
    ):
        """Initialize critic network.

        Args:
            state_dim: Dimension of state input.
            action_dim: Dimension of action input.
            hidden_sizes: Hidden layer sizes.
            activation: Activation function.
        """
        super().__init__()

        self.q_net = create_mlp(
            input_dim=state_dim + action_dim,
            output_dim=1,
            hidden_sizes=hidden_sizes,
            activation=activation,
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute Q-value.

        Args:
            state: State tensor (batch_size, state_dim).
            action: Action tensor (batch_size, action_dim).

        Returns:
            Q-value tensor (batch_size, 1).
        """
        x = torch.cat([state, action], dim=-1)
        return self.q_net(x)


class TwinCritic(nn.Module):
    """Twin Q-networks for SAC.

    Uses two separate Q-networks and takes the minimum for
    target value computation to reduce overestimation bias.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...] = (64, 64),
        activation: str = "relu",
    ):
        """Initialize twin critics.

        Args:
            state_dim: Dimension of state input.
            action_dim: Dimension of action input.
            hidden_sizes: Hidden layer sizes.
            activation: Activation function.
        """
        super().__init__()

        self.q1 = Critic(state_dim, action_dim, hidden_sizes, activation)
        self.q2 = Critic(state_dim, action_dim, hidden_sizes, activation)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute both Q-values.

        Args:
            state: State tensor (batch_size, state_dim).
            action: Action tensor (batch_size, action_dim).

        Returns:
            Tuple of (q1_value, q2_value).
        """
        q1 = self.q1(state, action)
        q2 = self.q2(state, action)
        return q1, q2

    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute only Q1 value (for policy update)."""
        return self.q1(state, action)

    def min_q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute minimum of Q1 and Q2."""
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Soft update target network parameters.

    target = tau * source + (1 - tau) * target

    Args:
        target: Target network to update.
        source: Source network.
        tau: Interpolation coefficient (0 < tau <= 1).
    """
    for target_param, source_param in zip(
        target.parameters(), source.parameters(), strict=False
    ):
        target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    """Hard update: copy all parameters from source to target."""
    target.load_state_dict(source.state_dict())

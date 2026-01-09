"""Replay buffer for SAC training."""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


class Transition(NamedTuple):
    """A single transition tuple."""

    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool


@dataclass
class BatchSample:
    """A batch of transitions for training."""

    states: np.ndarray  # (batch_size, state_dim)
    actions: np.ndarray  # (batch_size, action_dim)
    rewards: np.ndarray  # (batch_size,)
    next_states: np.ndarray  # (batch_size, state_dim)
    dones: np.ndarray  # (batch_size,)


class ReplayBuffer:
    """Experience replay buffer for off-policy SAC training.

    Stores transitions (s, a, r, s', done) and allows random sampling
    for mini-batch gradient updates.
    """

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        seed: int = 42,
    ):
        """Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store.
            state_dim: Dimension of state vectors.
            action_dim: Dimension of action vectors.
            seed: Random seed for sampling.
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Pre-allocate arrays for efficiency
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self.ptr = 0  # Current write position
        self.size = 0  # Number of transitions stored

        self.rng = np.random.default_rng(seed)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition to the buffer.

        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Whether episode terminated.
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> BatchSample:
        """Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            BatchSample containing the sampled transitions.
        """
        indices = self.rng.integers(0, self.size, size=batch_size)

        return BatchSample(
            states=self.states[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_states=self.next_states[indices],
            dones=self.dones[indices],
        )

    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return self.size >= batch_size

    def clear(self) -> None:
        """Clear the buffer."""
        self.ptr = 0
        self.size = 0

    def add_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """Add multiple transitions at once.

        Args:
            states: Array of states (n, state_dim).
            actions: Array of actions (n, action_dim).
            rewards: Array of rewards (n,).
            next_states: Array of next states (n, state_dim).
            dones: Array of done flags (n,).
        """
        n = len(states)
        for i in range(n):
            self.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

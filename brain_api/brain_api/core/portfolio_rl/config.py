"""PPO configuration for portfolio RL."""

from dataclasses import dataclass
from typing import Any


@dataclass
class PPOConfig:
    """PPO hyperparameters and environment configuration.

    This config is used by both ppo_lstm and ppo_patchtst variants.
    The only difference between variants is the forecast feature source,
    which is injected at runtime.
    """

    # === Policy network ===
    hidden_sizes: tuple[int, ...] = (64, 64)
    activation: str = "tanh"

    # === PPO algorithm ===
    learning_rate: float = 3e-4
    clip_epsilon: float = 0.2
    gae_lambda: float = 0.95
    gamma: float = 0.99  # discount factor
    entropy_coef: float = 0.01  # entropy bonus for exploration
    value_coef: float = 0.5  # value function loss coefficient
    max_grad_norm: float = 0.5  # gradient clipping

    # === Training ===
    n_epochs: int = 10  # epochs per rollout batch
    batch_size: int = 64
    rollout_steps: int = 52  # ~1 year of weekly steps per rollout
    total_timesteps: int = 10_000  # total training steps

    # === Environment ===
    cost_bps: int = 10  # transaction cost in basis points (0.10%)
    cash_buffer: float = 0.02  # minimum cash weight (2%)
    max_position_weight: float = 0.20  # max weight per stock (20%)
    reward_scale: float = 100.0  # multiply returns by this (1% → 1.0)

    # === Universe ===
    n_stocks: int = 15  # Top-15 stocks by liquidity
    # Action space = n_stocks + 1 (for CASH)

    # === Reproducibility ===
    seed: int = 42

    # === Evaluation ===
    validation_years: int = 2  # years to hold out for validation
    min_sharpe_improvement: float = 0.0  # must beat baseline by this margin

    @property
    def action_dim(self) -> int:
        """Action dimension = n_stocks + CASH."""
        return self.n_stocks + 1

    @property
    def cost_rate(self) -> float:
        """Convert basis points to decimal rate."""
        return self.cost_bps / 10_000

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "hidden_sizes": list(self.hidden_sizes),
            "activation": self.activation,
            "learning_rate": self.learning_rate,
            "clip_epsilon": self.clip_epsilon,
            "gae_lambda": self.gae_lambda,
            "gamma": self.gamma,
            "entropy_coef": self.entropy_coef,
            "value_coef": self.value_coef,
            "max_grad_norm": self.max_grad_norm,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "rollout_steps": self.rollout_steps,
            "total_timesteps": self.total_timesteps,
            "cost_bps": self.cost_bps,
            "cash_buffer": self.cash_buffer,
            "max_position_weight": self.max_position_weight,
            "reward_scale": self.reward_scale,
            "n_stocks": self.n_stocks,
            "seed": self.seed,
            "validation_years": self.validation_years,
            "min_sharpe_improvement": self.min_sharpe_improvement,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PPOConfig":
        """Create config from dictionary."""
        # Handle hidden_sizes conversion from list to tuple
        if "hidden_sizes" in data and isinstance(data["hidden_sizes"], list):
            data = data.copy()
            data["hidden_sizes"] = tuple(data["hidden_sizes"])
        return cls(**data)


# Default configuration
DEFAULT_PPO_CONFIG = PPOConfig()

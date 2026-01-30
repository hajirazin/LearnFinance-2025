"""PPO configuration with dual forecasts (LSTM + PatchTST).

Extends the shared PPOBaseConfig with training-specific settings.
"""

from dataclasses import dataclass
from typing import Any

from brain_api.core.portfolio_rl.config import PPOBaseConfig


@dataclass
class PPOConfig(PPOBaseConfig):
    """Configuration for unified PPO agent with dual forecasts.

    Inherits all PPOBaseConfig settings and adds training-specific ones.
    The agent receives both LSTM and PatchTST forecasts as input features.
    """

    # Training data lookback (years of historical data)
    training_years: int = 10

    # Walk-forward evaluation settings
    n_eval_folds: int = 3  # number of expanding-window folds

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "training_years": self.training_years,
                "n_eval_folds": self.n_eval_folds,
            }
        )
        return base_dict

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

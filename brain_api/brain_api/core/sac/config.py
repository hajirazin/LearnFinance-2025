"""SAC configuration with dual forecasts (LSTM + PatchTST).

Extends the shared SACBaseConfig with training-specific settings.
"""

from dataclasses import dataclass
from typing import Any

from brain_api.core.portfolio_rl.sac_config import SACBaseConfig


@dataclass
class SACConfig(SACBaseConfig):
    """Configuration for unified SAC agent with dual forecasts.

    Inherits all SACBaseConfig settings and adds training-specific ones.
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
    def from_dict(cls, data: dict[str, Any]) -> "SACConfig":
        """Create config from dictionary."""
        if "hidden_sizes" in data and isinstance(data["hidden_sizes"], list):
            data = data.copy()
            data["hidden_sizes"] = tuple(data["hidden_sizes"])
        return cls(**data)


# Default configuration
DEFAULT_SAC_CONFIG = SACConfig()

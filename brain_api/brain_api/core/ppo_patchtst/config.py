"""PPO + PatchTST configuration.

Extends the shared PPOConfig with PatchTST-specific settings.
"""

from dataclasses import dataclass
from typing import Any

from brain_api.core.portfolio_rl.config import PPOConfig


@dataclass
class PPOPatchTSTConfig(PPOConfig):
    """Configuration for PPO + PatchTST variant.

    Inherits all PPOConfig settings and adds PatchTST-specific ones.
    The forecast feature comes from a pre-trained PatchTST model.
    """

    # PatchTST model version to use for forecast features
    # If None, uses the current promoted PatchTST version
    patchtst_version: str | None = None

    # Training data lookback (years of historical data)
    training_years: int = 10

    # Walk-forward evaluation settings
    n_eval_folds: int = 3  # number of expanding-window folds

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            "patchtst_version": self.patchtst_version,
            "training_years": self.training_years,
            "n_eval_folds": self.n_eval_folds,
        })
        return base_dict

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PPOPatchTSTConfig":
        """Create config from dictionary."""
        if "hidden_sizes" in data and isinstance(data["hidden_sizes"], list):
            data = data.copy()
            data["hidden_sizes"] = tuple(data["hidden_sizes"])
        return cls(**data)


# Default configuration
DEFAULT_PPO_PATCHTST_CONFIG = PPOPatchTSTConfig()


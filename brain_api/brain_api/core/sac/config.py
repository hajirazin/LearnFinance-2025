"""SAC configuration with dual forecasts (LSTM + PatchTST).

Extends the shared SACBaseConfig with training-specific settings.
"""

from dataclasses import dataclass, replace
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


def make_sac_config_for_n_stocks(base: SACConfig, n_stocks: int) -> SACConfig:
    """Return a copy of ``base`` resized for ``n_stocks`` risky assets.

    Thin SACConfig-layer wrapper around
    :func:`brain_api.core.portfolio_rl.sac_config.make_sac_base_config_for_n_stocks`
    that preserves the SACConfig-only fields (``training_years``,
    ``n_eval_folds``). The endpoint calls this factory after resolving
    the bucket symbol list so each parallel A/B bucket trains its SAC
    actor/critic at the right action dimension without mutating any
    shared global config.

    See the base helper for the math contract (``target_entropy =
    -(n_stocks + 1)``) and the byte-equivalence guarantee at
    ``n_stocks == base.n_stocks``.
    """
    if n_stocks < 1:
        raise ValueError(
            f"n_stocks must be >= 1 to build a meaningful SAC action space, "
            f"got {n_stocks}."
        )
    return replace(
        base,
        n_stocks=n_stocks,
        target_entropy=-float(n_stocks + 1),
    )


# Default configuration
DEFAULT_SAC_CONFIG = SACConfig()

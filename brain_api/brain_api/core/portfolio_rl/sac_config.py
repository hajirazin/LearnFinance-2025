"""SAC base configuration for portfolio RL."""

from dataclasses import dataclass, field, replace
from typing import Any

from brain_api.core.portfolio_rl.broker_costs import IBKRSingaporeCostConfig


@dataclass
class SACBaseConfig:
    """SAC hyperparameters optimized for weekly portfolio RL with limited data.

    This is the base config with shared settings for all SAC variants.
    Variant-specific configs (SACConfig) extend this.
    """

    # === Networks (smaller due to limited data ~500 transitions) ===
    hidden_sizes: tuple[int, ...] = (64, 64)
    activation: str = "relu"  # ReLU is standard for SAC

    # === SAC algorithm ===
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4  # For auto-entropy tuning
    tau: float = 0.005  # Target network Polyak update rate
    gamma: float = 0.97  # Weekly steps: 1/(1-0.97) ≈ 33 weeks (~8-month horizon)

    # === Entropy tuning ===
    # For a tanh-squashed Gaussian policy, the standard SAC heuristic is
    # target_entropy = -dim(action).
    # For 16 dims (15 stocks + cash), -16.0 encourages moderate exploration.
    auto_entropy_tuning: bool = True
    target_entropy: float | None = -16.0  # target_entropy = -dim(action)
    init_alpha: float = 0.2  # Moderate initial entropy coefficient

    # === Training ===
    buffer_size: int = 10_000  # More than enough for weekly data
    batch_size: int = 64  # Smaller batch for limited data
    gradient_steps_per_env_step: int = 1
    warmup_steps: int = 100  # Random actions before training starts
    total_timesteps: int = 10_000

    # === Regularization (for limited data) ===
    weight_decay: float = 1e-4  # L2 regularization
    max_grad_norm: float = 1.0  # Gradient clipping
    q_value_clip: float = 100.0  # Clip Q-targets to prevent divergence
    normalize_rewards: bool = True  # Use running reward normalization
    hhi_penalty_scale: float = 0.4  # Soft penalty for portfolio concentration

    # === Environment (shared RL environment defaults) ===
    # ``cost_bps`` is **deprecated** -- the live cost source is the
    # IBKR Singapore Tiered model on ``cost_config``. Retained for
    # round-trip compatibility with previously serialised configs.
    cost_bps: int = 10  # DEPRECATED; see cost_config
    cash_buffer: float = 0.02  # Minimum cash weight (2%)
    max_position_weight: float = 0.20  # Max weight per stock (20%)
    reward_scale: float = 1.0  # Let normalize_rewards handle magnitude.
    # SAC paper: alpha ≡ 1/reward_scale. Having reward_scale=100
    # AND normalize_rewards AND auto_entropy_tuning creates 3 competing
    # magnitude controls. With reward_scale=1.0, Welford normalization
    # produces mean~0 std~1 rewards, giving alpha a stable target.
    n_stocks: int = 15  # Top-15 stocks by liquidity
    # IBKR Singapore Tiered transaction-cost schedule. See
    # brain_api/core/portfolio_rl/broker_costs.py for the per-leg math.
    cost_config: IBKRSingaporeCostConfig = field(
        default_factory=IBKRSingaporeCostConfig.default
    )

    # === Reward shaping ===
    sharpe_weight: float = (
        0.5  # Blend: sharpe_weight * DSR + (1-sharpe_weight) * return_reward
    )
    sharpe_eta: float = 0.01  # EMA decay for differential Sharpe (~100-week window)

    # === Reproducibility ===
    seed: int = 42

    # === Evaluation ===
    validation_years: int = 2
    min_cagr_improvement: float = 0.0  # Must beat baseline by this margin

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
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr,
            "alpha_lr": self.alpha_lr,
            "tau": self.tau,
            "gamma": self.gamma,
            "auto_entropy_tuning": self.auto_entropy_tuning,
            "target_entropy": self.target_entropy,
            "init_alpha": self.init_alpha,
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "gradient_steps_per_env_step": self.gradient_steps_per_env_step,
            "warmup_steps": self.warmup_steps,
            "total_timesteps": self.total_timesteps,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "q_value_clip": self.q_value_clip,
            "normalize_rewards": self.normalize_rewards,
            "hhi_penalty_scale": self.hhi_penalty_scale,
            "cost_bps": self.cost_bps,
            "cash_buffer": self.cash_buffer,
            "max_position_weight": self.max_position_weight,
            "reward_scale": self.reward_scale,
            "n_stocks": self.n_stocks,
            "seed": self.seed,
            "validation_years": self.validation_years,
            "min_cagr_improvement": self.min_cagr_improvement,
            "sharpe_weight": self.sharpe_weight,
            "sharpe_eta": self.sharpe_eta,
            "cost_config": self.cost_config.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SACBaseConfig":
        """Create config from dictionary."""
        data = data.copy()
        if "hidden_sizes" in data and isinstance(data["hidden_sizes"], list):
            data["hidden_sizes"] = tuple(data["hidden_sizes"])
        # Round-trip the IBKR cost sub-config; legacy serialised configs
        # without the field fall through to the dataclass default factory.
        if "cost_config" in data and isinstance(data["cost_config"], dict):
            data["cost_config"] = IBKRSingaporeCostConfig.from_dict(data["cost_config"])
        return cls(**data)


def make_sac_base_config_for_n_stocks(
    base: SACBaseConfig, n_stocks: int
) -> SACBaseConfig:
    """Return a copy of ``base`` with action-dim-sensitive fields rewritten.

    The SAC paper sets ``target_entropy = -dim(action)`` as the textbook
    default for continuous control with squashed Gaussian actions.
    The ``-(n_stocks + 1)`` accounts for the CASH slot (`SACBaseConfig.action_dim`
    is ``n_stocks + 1``). Two parallel A/B buckets (``halal_filtered``
    fixed at 15, ``halal`` variable size from yfinance ETF top-holdings)
    therefore need different ``target_entropy`` values, and we cannot
    mutate the global ``DEFAULT_SAC_BASE_CONFIG`` in place because both
    Sunday training jobs share the same FastAPI process.

    Returns a fresh dataclass instance so the global default stays
    immutable. Every other hyperparameter (``hidden_sizes``, ``gamma``,
    ``tau``, ``actor_lr``, ...) is inherited verbatim from ``base`` so
    research-driven SAC settings are not silently re-tuned per bucket.

    Math invariant: when ``n_stocks == base.n_stocks`` (currently 15 ==
    15 for the halal_filtered bucket), the returned config is byte-
    equivalent to ``base`` (``target_entropy`` ends up at the same
    -16.0). This preserves halal_filtered's existing ``compute_version``
    hash and ``current`` artifact lineage.

    Args:
        base: Source config to copy from (typically
            ``DEFAULT_SAC_BASE_CONFIG``).
        n_stocks: Bucket-determined number of risky assets in the
            action vector (cash is added on top).

    Returns:
        A new ``SACBaseConfig`` instance with ``n_stocks`` and
        ``target_entropy`` rewritten.
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


@dataclass
class SACFinetuneConfig:
    """Configuration for weekly SAC fine-tuning."""

    lookback_weeks: int = 26  # 6-month rolling buffer
    total_timesteps: int = 2_000  # Much smaller than full training
    actor_lr: float = 1e-4  # Lower LR for fine-tuning
    critic_lr: float = 1e-4
    alpha_lr: float = 1e-4


# Default configuration
DEFAULT_SAC_BASE_CONFIG = SACBaseConfig()

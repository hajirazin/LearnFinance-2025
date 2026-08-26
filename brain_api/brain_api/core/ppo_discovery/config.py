"""Locked hyperparameters and feature contracts for ppo_discovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MAX_ASSETS = 512
MIN_ELIGIBLE_ASSETS = 10
MAX_SELECTED = 15
CASH_FLOOR = 0.02
HISTORY_BARS = 253
ENCODER_SESSIONS = 250
ENCODER_CHANNELS = 4
PATCH_LENGTH = 5
PATCH_STRIDE = 5
N_PATCHES = ENCODER_SESSIONS // PATCH_LENGTH
TEMPORAL_D_MODEL = 64
SET_D_MODEL = 128
EXPLICIT_ASSET_FEATURES = 9
GLOBAL_FEATURES = 7
TOKEN_WIDTH = TEMPORAL_D_MODEL + EXPLICIT_ASSET_FEATURES
PROMOTION_CAGR_FLOOR = 0.12
UNIVERSE_NAME = "halal_new"
MODEL_TYPE = "ppo_discovery"
ALGORITHM = "ppo_discovery"
DECISION_TIMEZONE = "America/New_York"
DECISION_HOUR = 9
NEWS_RECENCY_TAU_HOURS = 168.0
ARTICLE_PAGE_CAP = 10_000

ASSET_FEATURE_NAMES: tuple[str, ...] = (
    "momentum_1w_cs_rank",
    "momentum_4w_cs_rank",
    "momentum_12_1_cs_rank",
    "realized_vol_20d_cs_rank",
    "news_sentiment_cs_rank",
    "raw_news_sentiment",
    "log1p_article_count",
    "news_recency",
    "current_weight",
)
GLOBAL_FEATURE_NAMES: tuple[str, ...] = (
    "p_calm",
    "p_stress",
    "cash_weight",
    "fraction_momentum_4w_positive",
    "spy_return_20d",
    "median_raw_news_sentiment",
    "fraction_of_assets_with_news",
)
AUDIT_NEWS_FIELDS: tuple[str, ...] = (
    "average_confidence",
    "sentiment_dispersion",
    "unique_source_count",
    "has_news",
    "fraction_negative_news",
    "fraction_positive_news",
    "news_sentiment_dispersion",
)
EXPERIMENT_SEEDS: tuple[int, ...] = (42, 123, 2026, 7, 19, 31, 73, 101, 211, 509)
REQUIRED_ABLATIONS: tuple[str, ...] = (
    "full_ppo",
    "no_news_features",
    "news_time_shuffled",
    "no_temporal_encoder",
    "frozen_pretrained_encoder",
    "fixed_k_15",
    "equal_weight_selected",
    "no_hmm_globals",
    "no_transaction_cost_term",
    "no_supervised_pretraining",
)

if len(ASSET_FEATURE_NAMES) != EXPLICIT_ASSET_FEATURES:
    raise RuntimeError("ASSET_FEATURE_NAMES length must match EXPLICIT_ASSET_FEATURES")
if len(GLOBAL_FEATURE_NAMES) != GLOBAL_FEATURES:
    raise RuntimeError("GLOBAL_FEATURE_NAMES length must match GLOBAL_FEATURES")


@dataclass
class PPODiscoveryConfig:
    """Versioned training and architecture hyperparameters."""

    max_assets: int = MAX_ASSETS
    min_eligible_assets: int = MIN_ELIGIBLE_ASSETS
    max_selected: int = MAX_SELECTED
    cash_floor: float = CASH_FLOOR
    history_bars: int = HISTORY_BARS
    encoder_sessions: int = ENCODER_SESSIONS
    encoder_channels: int = ENCODER_CHANNELS
    patch_length: int = PATCH_LENGTH
    patch_stride: int = PATCH_STRIDE
    temporal_d_model: int = TEMPORAL_D_MODEL
    temporal_heads: int = 4
    temporal_layers: int = 2
    temporal_ff: int = 128
    set_d_model: int = SET_D_MODEL
    set_heads: int = 4
    set_layers: int = 2
    set_ff: int = 256
    dropout: float = 0.10
    gamma: float = 0.97
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.20
    value_loss_coef: float = 0.50
    count_entropy_coef: float = 0.01
    selection_entropy_coef: float = 0.01
    actor_lr: float = 3e-4
    encoder_finetune_lr: float = 1e-5
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    rollout_length: int = 52
    minibatch_size: int = 32
    ppo_epochs: int = 5
    freeze_encoder_updates: int = 20
    total_timesteps: int = 10_000
    pretrain_lr: float = 3e-4
    pretrain_batch_size: int = 256
    pretrain_max_epochs: int = 50
    pretrain_patience: int = 8
    pretrain_huber_beta: float = 0.01
    hhi_penalty_scale: float = 0.4
    reward_scale: float = 1.0
    seeds: tuple[int, ...] = EXPERIMENT_SEEDS
    universe: str = UNIVERSE_NAME

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "max_assets": self.max_assets,
            "min_eligible_assets": self.min_eligible_assets,
            "max_selected": self.max_selected,
            "cash_floor": self.cash_floor,
            "history_bars": self.history_bars,
            "encoder_sessions": self.encoder_sessions,
            "encoder_channels": self.encoder_channels,
            "patch_length": self.patch_length,
            "patch_stride": self.patch_stride,
            "temporal_d_model": self.temporal_d_model,
            "temporal_heads": self.temporal_heads,
            "temporal_layers": self.temporal_layers,
            "temporal_ff": self.temporal_ff,
            "set_d_model": self.set_d_model,
            "set_heads": self.set_heads,
            "set_layers": self.set_layers,
            "set_ff": self.set_ff,
            "dropout": self.dropout,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_epsilon": self.clip_epsilon,
            "value_loss_coef": self.value_loss_coef,
            "count_entropy_coef": self.count_entropy_coef,
            "selection_entropy_coef": self.selection_entropy_coef,
            "actor_lr": self.actor_lr,
            "encoder_finetune_lr": self.encoder_finetune_lr,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "rollout_length": self.rollout_length,
            "minibatch_size": self.minibatch_size,
            "ppo_epochs": self.ppo_epochs,
            "freeze_encoder_updates": self.freeze_encoder_updates,
            "total_timesteps": self.total_timesteps,
            "pretrain_lr": self.pretrain_lr,
            "pretrain_batch_size": self.pretrain_batch_size,
            "pretrain_max_epochs": self.pretrain_max_epochs,
            "pretrain_patience": self.pretrain_patience,
            "pretrain_huber_beta": self.pretrain_huber_beta,
            "hhi_penalty_scale": self.hhi_penalty_scale,
            "reward_scale": self.reward_scale,
            "seeds": list(self.seeds),
            "universe": self.universe,
            "asset_feature_names": list(ASSET_FEATURE_NAMES),
            "global_feature_names": list(GLOBAL_FEATURE_NAMES),
        }
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PPODiscoveryConfig:
        payload = data.copy()
        payload.pop("asset_feature_names", None)
        payload.pop("global_feature_names", None)
        if "seeds" in payload and isinstance(payload["seeds"], list):
            payload["seeds"] = tuple(payload["seeds"])
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in payload.items() if key in allowed})


DEFAULT_PPO_DISCOVERY_CONFIG = PPODiscoveryConfig()


@dataclass
class TrainingConfig:
    """Runtime training knobs; defaults match the research protocol."""

    ppo: PPODiscoveryConfig = field(default_factory=PPODiscoveryConfig)
    experiment_id: str = "ppo-discovery-default"
    end_date: str | None = None


__all__ = [
    "ALGORITHM",
    "ARTICLE_PAGE_CAP",
    "ASSET_FEATURE_NAMES",
    "AUDIT_NEWS_FIELDS",
    "CASH_FLOOR",
    "DECISION_HOUR",
    "DECISION_TIMEZONE",
    "DEFAULT_PPO_DISCOVERY_CONFIG",
    "ENCODER_CHANNELS",
    "ENCODER_SESSIONS",
    "EXPERIMENT_SEEDS",
    "EXPLICIT_ASSET_FEATURES",
    "GLOBAL_FEATURES",
    "GLOBAL_FEATURE_NAMES",
    "HISTORY_BARS",
    "MAX_ASSETS",
    "MAX_SELECTED",
    "MIN_ELIGIBLE_ASSETS",
    "MODEL_TYPE",
    "NEWS_RECENCY_TAU_HOURS",
    "N_PATCHES",
    "PATCH_LENGTH",
    "PATCH_STRIDE",
    "PROMOTION_CAGR_FLOOR",
    "REQUIRED_ABLATIONS",
    "SET_D_MODEL",
    "TEMPORAL_D_MODEL",
    "TOKEN_WIDTH",
    "UNIVERSE_NAME",
    "PPODiscoveryConfig",
    "TrainingConfig",
]

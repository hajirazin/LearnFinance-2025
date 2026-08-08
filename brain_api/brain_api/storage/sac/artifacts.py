"""Loaded SAC artifact aggregate."""

from dataclasses import dataclass

import torch

from brain_api.core.portfolio_rl.sac_networks import GaussianActor, TwinCritic
from brain_api.core.portfolio_rl.scaler import PortfolioScaler
from brain_api.core.sac.config import SACConfig


@dataclass
class SACArtifacts:
    """Everything required for deterministic SAC inference."""

    config: SACConfig
    scaler: PortfolioScaler
    actor: GaussianActor
    critic: TwinCritic
    critic_target: TwinCritic
    log_alpha: torch.Tensor
    symbol_order: list[str]
    version: str

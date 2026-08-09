"""Loaded SAC v3 artifact aggregate and manifest validation."""

from dataclasses import dataclass
from typing import Any

import torch

from brain_api.core.portfolio_rl.sac_networks import GaussianActor, TwinCritic
from brain_api.core.portfolio_rl.scaler import PortfolioScaler
from brain_api.core.portfolio_rl.state import (
    SAC_ASSET_FEATURE_NAMES,
    SAC_GLOBAL_FEATURE_NAMES,
    STATE_DIM,
)
from brain_api.core.sac.config import SACConfig
from brain_api.core.sac.regime_hmm import RegimeHMMArtifact

SAC_SCHEMA_VERSION = 3
SAC_ARCHITECTURE = "masked_attention"
SAC_MAX_ASSETS = 30
SAC_ACTION_DIM = 31


class SACArtifactCompatibilityError(ValueError):
    """Raised before loading weights from a legacy or incomplete artifact."""


@dataclass(frozen=True)
class SACV3AuxiliaryArtifacts:
    """Serializable non-network state required for stateless SAC v3 inference."""

    regime_hmm: RegimeHMMArtifact
    median_patchtst_scaler: dict[str, float]
    audit_metadata: dict[str, Any]

    @property
    def training_cutoff_date(self) -> str:
        """Cutoff exposed to orchestration before it requests live history."""
        return self.regime_hmm.training_cutoff_date.isoformat()

    @property
    def training_cutoff_posterior(self) -> tuple[float, float, float]:
        """Persisted filtered posterior used to continue stateless inference."""
        values = self.regime_hmm.terminal_posterior
        return float(values[0]), float(values[1]), float(values[2])

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON representation stored with the model."""
        return {
            "regime_hmm": self.regime_hmm.to_dict(),
            "median_patchtst_scaler": self.median_patchtst_scaler,
            "audit_metadata": self.audit_metadata,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SACV3AuxiliaryArtifacts":
        """Validate and load the non-network portion of a SAC v3 artifact."""
        required = {
            "regime_hmm",
            "median_patchtst_scaler",
            "audit_metadata",
        }
        missing = sorted(required - set(value))
        if missing:
            raise SACArtifactCompatibilityError(
                f"SAC v3 auxiliary artifact is incomplete; missing {missing}"
            )
        return cls(
            regime_hmm=RegimeHMMArtifact.from_dict(dict(value["regime_hmm"])),
            median_patchtst_scaler={
                str(key): float(item)
                for key, item in value["median_patchtst_scaler"].items()
            },
            audit_metadata=dict(value["audit_metadata"]),
        )


def validate_sac_v3_metadata(metadata: dict[str, Any], symbol_order: list[str]) -> None:
    """Reject legacy metadata and inconsistent slot maps before weight loading."""
    if metadata.get("sac_schema_version") != SAC_SCHEMA_VERSION:
        raise SACArtifactCompatibilityError(
            "Legacy SAC artifact rejected: sac_schema_version must be 3; "
            "retrain this universe with SAC v3"
        )
    if metadata.get("architecture") != SAC_ARCHITECTURE:
        raise SACArtifactCompatibilityError(
            "Incompatible SAC artifact architecture; expected masked_attention"
        )
    if metadata.get("max_assets") != SAC_MAX_ASSETS:
        raise SACArtifactCompatibilityError("SAC v3 max_assets must be 30")
    if metadata.get("action_dim") != SAC_ACTION_DIM:
        raise SACArtifactCompatibilityError("SAC v3 action_dim must be 31")
    if metadata.get("state_dim") != STATE_DIM:
        raise SACArtifactCompatibilityError("SAC v3 state_dim must be 245")
    if metadata.get("asset_feature_names") != list(SAC_ASSET_FEATURE_NAMES):
        raise SACArtifactCompatibilityError(
            "SAC v3 asset feature manifest does not match the runtime schema"
        )
    if metadata.get("global_feature_names") != list(SAC_GLOBAL_FEATURE_NAMES):
        raise SACArtifactCompatibilityError(
            "SAC v3 global feature manifest does not match the runtime schema"
        )
    if symbol_order != sorted(symbol_order) or len(set(symbol_order)) != len(
        symbol_order
    ):
        raise SACArtifactCompatibilityError(
            "SAC v3 symbol_order must be unique and lexicographically sorted"
        )
    if not 1 <= len(symbol_order) <= SAC_MAX_ASSETS:
        raise SACArtifactCompatibilityError(
            "SAC v3 symbol_order must contain between 1 and 30 symbols"
        )
    expected_slots = {symbol: slot for slot, symbol in enumerate(symbol_order)}
    if metadata.get("symbols") != symbol_order:
        raise SACArtifactCompatibilityError(
            "SAC v3 metadata symbols do not match canonical symbol_order"
        )
    if metadata.get("symbol_to_slot") != expected_slots:
        raise SACArtifactCompatibilityError(
            "SAC v3 symbol_to_slot does not match canonical symbol_order"
        )


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
    metadata: dict[str, Any]
    v3_auxiliary: SACV3AuxiliaryArtifacts

"""Shared filesystem implementation for universe-keyed SAC buckets."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch

from brain_api.core.portfolio_rl.sac_networks import GaussianActor, TwinCritic
from brain_api.core.portfolio_rl.scaler import PortfolioScaler
from brain_api.core.sac.config import SACConfig
from brain_api.storage.base import DEFAULT_DATA_PATH

from .artifacts import (
    SACArtifactCompatibilityError,
    SACArtifacts,
    SACV3AuxiliaryArtifacts,
    validate_sac_v3_metadata,
)

_V3_AUXILIARY_FILE = "sac_v3_auxiliary.json"
_V3_REQUIRED_FILES = frozenset(
    {
        "actor.pt",
        "critic.pt",
        "critic_target.pt",
        "log_alpha.pt",
        "scaler.pkl",
        "config.json",
        "symbol_order.json",
        "metadata.json",
        _V3_AUXILIARY_FILE,
    }
)


def _state_shapes(module: torch.nn.Module) -> dict[str, tuple[int, ...]]:
    return {name: tuple(value.shape) for name, value in module.state_dict().items()}


class SACFilesystemStorage:
    """Store one SAC universe bucket with an independent current pointer."""

    bucket_name: str

    def __init__(self, base_path: Path | str | None = None):
        self.base_path = Path(base_path or DEFAULT_DATA_PATH)
        self._model_path = self.base_path / "models" / self.bucket_name

    @property
    def model_type(self) -> str:
        """Return the universe-keyed bucket identifier."""
        return self.bucket_name

    def _version_path(self, version: str) -> Path:
        return self._model_path / version

    def _artifact_path(self, version: str, candidate_seed: int | None = None) -> Path:
        version_path = self._version_path(version)
        if candidate_seed is None:
            return version_path
        return version_path / "candidates" / f"seed-{candidate_seed}"

    def version_exists(self, version: str) -> bool:
        """Return whether the complete canonical SAC v3 version exists."""
        version_path = self._version_path(version)
        return all((version_path / name).is_file() for name in _V3_REQUIRED_FILES)

    def write_artifacts(
        self,
        version: str,
        actor: GaussianActor,
        critic: TwinCritic,
        critic_target: TwinCritic,
        log_alpha: torch.Tensor,
        scaler: PortfolioScaler,
        config: SACConfig,
        symbol_order: list[str],
        metadata: dict[str, Any],
        v3_auxiliary: SACV3AuxiliaryArtifacts,
    ) -> Path:
        """Write canonical artifacts without changing the current pointer."""
        return self._write_artifacts_to(
            self._artifact_path(version),
            actor,
            critic,
            critic_target,
            log_alpha,
            scaler,
            config,
            symbol_order,
            metadata,
            v3_auxiliary,
        )

    def write_candidate_artifacts(
        self,
        version: str,
        seed: int,
        actor: GaussianActor,
        critic: TwinCritic,
        critic_target: TwinCritic,
        log_alpha: torch.Tensor,
        scaler: PortfolioScaler,
        config: SACConfig,
        symbol_order: list[str],
        metadata: dict[str, Any],
        v3_auxiliary: SACV3AuxiliaryArtifacts,
    ) -> Path:
        """Write one fixed-seed candidate below ``candidates/seed-N``."""
        return self._write_artifacts_to(
            self._artifact_path(version, seed),
            actor,
            critic,
            critic_target,
            log_alpha,
            scaler,
            config,
            symbol_order,
            metadata,
            v3_auxiliary,
        )

    def write_candidate_metadata(
        self,
        version: str,
        seed: int,
        metadata: dict[str, Any],
    ) -> Path:
        """Update only a candidate's audit metadata after health evaluation."""
        candidate_dir = self._artifact_path(version, seed)
        if not (candidate_dir / "actor.pt").is_file():
            raise ValueError(
                f"Cannot update incomplete SAC candidate seed {seed} for {version}"
            )
        metadata_path = candidate_dir / "metadata.json"
        fd, temp_path = tempfile.mkstemp(
            dir=candidate_dir,
            prefix=".metadata_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write(
                    json.dumps(metadata, indent=2, default=str, allow_nan=False)
                )
            os.replace(temp_path, metadata_path)
        except Exception:
            with contextlib.suppress(OSError):
                os.close(fd)
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise
        return metadata_path

    @staticmethod
    def _write_artifacts_to(
        artifact_dir: Path,
        actor: GaussianActor,
        critic: TwinCritic,
        critic_target: TwinCritic,
        log_alpha: torch.Tensor,
        scaler: PortfolioScaler,
        config: SACConfig,
        symbol_order: list[str],
        metadata: dict[str, Any],
        v3_auxiliary: SACV3AuxiliaryArtifacts,
    ) -> Path:
        validate_sac_v3_metadata(metadata, symbol_order)
        # Round-trip validates HMM shapes/config before any artifact is written.
        SACV3AuxiliaryArtifacts.from_dict(v3_auxiliary.to_dict())
        expected_median_scaler = {
            "mean": float(scaler.median_mean),
            "scale": float(scaler.median_scale),
        }
        if v3_auxiliary.median_patchtst_scaler != expected_median_scaler:
            raise SACArtifactCompatibilityError(
                "SAC v3 median PatchTST scaler does not match scaler.pkl"
            )
        expected_actor = GaussianActor(
            hidden_sizes=config.hidden_sizes,
            activation=config.activation,
        )
        expected_critic = TwinCritic(
            hidden_sizes=config.hidden_sizes,
            activation=config.activation,
        )
        if _state_shapes(actor) != _state_shapes(expected_actor):
            raise SACArtifactCompatibilityError(
                "SAC v3 actor architecture does not match config.json"
            )
        expected_critic_shapes = _state_shapes(expected_critic)
        if (
            _state_shapes(critic) != expected_critic_shapes
            or _state_shapes(critic_target) != expected_critic_shapes
        ):
            raise SACArtifactCompatibilityError(
                "SAC v3 critic architecture does not match config.json"
            )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        torch.save(actor.state_dict(), artifact_dir / "actor.pt")
        torch.save(critic.state_dict(), artifact_dir / "critic.pt")
        torch.save(critic_target.state_dict(), artifact_dir / "critic_target.pt")
        torch.save(log_alpha, artifact_dir / "log_alpha.pt")
        scaler.save(artifact_dir / "scaler.pkl")
        (artifact_dir / "config.json").write_text(
            json.dumps(config.to_dict(), indent=2)
        )
        (artifact_dir / "symbol_order.json").write_text(
            json.dumps(symbol_order, indent=2)
        )
        (artifact_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str, allow_nan=False)
        )
        (artifact_dir / _V3_AUXILIARY_FILE).write_text(
            json.dumps(v3_auxiliary.to_dict(), indent=2, allow_nan=False)
        )
        return artifact_dir

    def promote_candidate(self, version: str, seed: int) -> Path:
        """Copy only the selected seed candidate into the canonical version root."""
        candidate_dir = self._artifact_path(version, seed)
        required = _V3_REQUIRED_FILES
        missing = sorted(
            name for name in required if not (candidate_dir / name).is_file()
        )
        if missing:
            raise ValueError(
                f"SAC candidate seed {seed} is incomplete for {version}: {missing}"
            )
        version_dir = self._version_path(version)
        version_dir.mkdir(parents=True, exist_ok=True)
        for name in required:
            shutil.copy2(candidate_dir / name, version_dir / name)
        return version_dir

    def read_current_version(self) -> str | None:
        current_file = self._model_path / "current"
        return current_file.read_text().strip() if current_file.exists() else None

    def read_metadata(self, version: str) -> dict[str, Any] | None:
        metadata_path = self._version_path(version) / "metadata.json"
        return json.loads(metadata_path.read_text()) if metadata_path.exists() else None

    def promote_version(self, version: str) -> None:
        """Atomically point this bucket at a fully written canonical version."""
        if not self.version_exists(version):
            raise ValueError(f"Cannot promote incomplete SAC version {version}")
        self._model_path.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(
            dir=self._model_path, prefix=".current_", suffix=".tmp"
        )
        try:
            os.write(fd, version.encode())
            os.close(fd)
            os.replace(temp_path, self._model_path / "current")
        except Exception:
            with contextlib.suppress(OSError):
                os.close(fd)
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    def load_config(self, version: str) -> SACConfig:
        return SACConfig.from_dict(
            json.loads((self._version_path(version) / "config.json").read_text())
        )

    def load_scaler(self, version: str) -> PortfolioScaler:
        return PortfolioScaler.load(self._version_path(version) / "scaler.pkl")

    def load_symbol_order(self, version: str) -> list[str]:
        return json.loads(
            (self._version_path(version) / "symbol_order.json").read_text()
        )

    def load_artifacts(self, version: str) -> SACArtifacts:
        """Load a complete SAC v3 artifact, rejecting legacy data first."""
        from brain_api.core.portfolio_rl.state import StateSchema

        version_dir = self._version_path(version)
        metadata_path = version_dir / "metadata.json"
        auxiliary_path = version_dir / _V3_AUXILIARY_FILE
        if not metadata_path.is_file():
            raise SACArtifactCompatibilityError(
                f"SAC artifact {version!r} has no metadata.json"
            )
        metadata = json.loads(metadata_path.read_text())
        symbol_order = self.load_symbol_order(version)
        validate_sac_v3_metadata(metadata, symbol_order)
        if not auxiliary_path.is_file():
            raise SACArtifactCompatibilityError(
                f"SAC v3 artifact {version!r} is missing {_V3_AUXILIARY_FILE}"
            )
        v3_auxiliary = SACV3AuxiliaryArtifacts.from_dict(
            json.loads(auxiliary_path.read_text())
        )
        config = self.load_config(version)
        scaler = self.load_scaler(version)
        if v3_auxiliary.median_patchtst_scaler != {
            "mean": float(scaler.median_mean),
            "scale": float(scaler.median_scale),
        }:
            raise SACArtifactCompatibilityError(
                "SAC v3 median PatchTST scaler conflicts with scaler.pkl"
            )
        schema = StateSchema()
        state_dim = schema.state_dim
        action_dim = schema.action_dim

        actor = GaussianActor(
            state_dim, action_dim, config.hidden_sizes, config.activation
        )
        actor.load_state_dict(
            torch.load(version_dir / "actor.pt", weights_only=True, map_location="cpu")
        )
        actor.eval()
        critic = TwinCritic(
            state_dim, action_dim, config.hidden_sizes, config.activation
        )
        critic.load_state_dict(
            torch.load(version_dir / "critic.pt", weights_only=True, map_location="cpu")
        )
        critic.eval()
        critic_target = TwinCritic(
            state_dim, action_dim, config.hidden_sizes, config.activation
        )
        critic_target.load_state_dict(
            torch.load(
                version_dir / "critic_target.pt",
                weights_only=True,
                map_location="cpu",
            )
        )
        critic_target.eval()
        log_alpha = torch.load(
            version_dir / "log_alpha.pt", weights_only=True, map_location="cpu"
        )
        return SACArtifacts(
            config=config,
            scaler=scaler,
            actor=actor,
            critic=critic,
            critic_target=critic_target,
            log_alpha=log_alpha,
            symbol_order=symbol_order,
            version=version,
            metadata=metadata,
            v3_auxiliary=v3_auxiliary,
        )

    def load_current_artifacts(self) -> SACArtifacts:
        version = self.read_current_version()
        if version is None:
            raise ValueError(f"No current SAC model version for {self.bucket_name}")
        return self.load_artifacts(version)

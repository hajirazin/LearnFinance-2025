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

from .artifacts import SACArtifacts


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
        """Return whether the canonical version directory exists."""
        return (self._version_path(version) / "actor.pt").is_file()

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
                handle.write(json.dumps(metadata, indent=2, default=str))
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
    ) -> Path:
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
            json.dumps(metadata, indent=2, default=str)
        )
        return artifact_dir

    def promote_candidate(self, version: str, seed: int) -> Path:
        """Copy only the selected seed candidate into the canonical version root."""
        candidate_dir = self._artifact_path(version, seed)
        required = {
            "actor.pt",
            "critic.pt",
            "critic_target.pt",
            "log_alpha.pt",
            "scaler.pkl",
            "config.json",
            "symbol_order.json",
            "metadata.json",
        }
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
        if not (self._version_path(version) / "actor.pt").is_file():
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
        """Load v1 or v2 artifacts; metadata absence means legacy schema v1."""
        config = self.load_config(version)
        scaler = self.load_scaler(version)
        symbol_order = self.load_symbol_order(version)
        metadata = self.read_metadata(version) or {}
        schema_version = int(metadata.get("state_schema_version", 1))
        if schema_version not in (1, 2):
            raise ValueError(f"Unsupported SAC state schema version {schema_version}")
        n_stocks = len(symbol_order)
        state_dim = (10 if schema_version == 1 else 11) * n_stocks + 1
        action_dim = n_stocks + 1
        version_dir = self._version_path(version)

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
            state_schema_version=schema_version,
        )

    def load_current_artifacts(self) -> SACArtifacts:
        version = self.read_current_version()
        if version is None:
            raise ValueError(f"No current SAC model version for {self.bucket_name}")
        return self.load_artifacts(version)

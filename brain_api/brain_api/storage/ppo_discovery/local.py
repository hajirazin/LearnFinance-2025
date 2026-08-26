"""Local filesystem storage for ppo_discovery artifacts."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from brain_api.core.ppo_discovery.config import PPODiscoveryConfig
from brain_api.storage.base import DEFAULT_DATA_PATH

REQUIRED_FILES: tuple[str, ...] = (
    "policy.pt",
    "pretrained_temporal_encoder.pt",
    "config.json",
    "feature_scalers.json",
    "regime_hmm.json",
    "metadata.json",
    "universe_manifest.json",
    "news_manifest.json",
    "price_manifest.json",
    "experiment_lock.json",
    "evaluation.json",
    "checksums.sha256",
)


@dataclass
class PPODiscoveryArtifacts:
    """Loaded promoted or candidate ppo_discovery artifact."""

    policy_state_dict: dict[str, torch.Tensor]
    config: PPODiscoveryConfig
    feature_scalers: dict[str, Any]
    regime_hmm: dict[str, Any]
    metadata: dict[str, Any]
    universe_manifest: dict[str, Any]
    version: str
    artifact_dir: Path


class PPODiscoveryHalalNewModelStorage:
    """Filesystem bucket ``ppo_discovery_halal_new`` with an independent current pointer."""

    bucket_name = "ppo_discovery_halal_new"

    def __init__(self, base_path: Path | str | None = None) -> None:
        self.base_path = Path(base_path or DEFAULT_DATA_PATH)
        self._model_path = self.base_path / "models" / self.bucket_name

    @property
    def model_type(self) -> str:
        return self.bucket_name

    def _version_path(self, version: str) -> Path:
        return self._model_path / version

    def version_exists(self, version: str) -> bool:
        version_path = self._version_path(version)
        return all((version_path / name).is_file() for name in REQUIRED_FILES)

    def read_current_version(self) -> str | None:
        current_file = self._model_path / "current"
        return current_file.read_text().strip() if current_file.exists() else None

    def write_artifacts(
        self,
        version: str,
        *,
        policy_state_dict: dict[str, Any],
        pretrained_encoder_state_dict: dict[str, Any],
        config: PPODiscoveryConfig,
        feature_scalers: dict[str, Any],
        regime_hmm: dict[str, Any],
        metadata: dict[str, Any],
        universe_manifest: dict[str, Any],
        news_manifest: dict[str, Any],
        price_manifest: dict[str, Any],
        experiment_lock: dict[str, Any],
        evaluation: dict[str, Any],
        promote: bool = False,
    ) -> Path:
        """Persist a candidate version. Never overwrites differing bytes."""
        version_path = self._version_path(version)
        if version_path.exists():
            existing = self.load_artifacts(version)
            if existing.metadata.get("config_hash") != metadata.get("config_hash"):
                raise ValueError(
                    f"refusing to overwrite ppo_discovery version {version} "
                    "with a different config hash"
                )
            return version_path
        version_path.mkdir(parents=True, exist_ok=True)
        torch.save(policy_state_dict, version_path / "policy.pt")
        torch.save(
            pretrained_encoder_state_dict,
            version_path / "pretrained_temporal_encoder.pt",
        )
        (version_path / "config.json").write_text(
            json.dumps(config.to_dict(), indent=2)
        )
        (version_path / "feature_scalers.json").write_text(
            json.dumps(feature_scalers, indent=2)
        )
        (version_path / "regime_hmm.json").write_text(json.dumps(regime_hmm, indent=2))
        (version_path / "metadata.json").write_text(json.dumps(metadata, indent=2))
        (version_path / "universe_manifest.json").write_text(
            json.dumps(universe_manifest, indent=2)
        )
        (version_path / "news_manifest.json").write_text(
            json.dumps(news_manifest, indent=2)
        )
        (version_path / "price_manifest.json").write_text(
            json.dumps(price_manifest, indent=2)
        )
        (version_path / "experiment_lock.json").write_text(
            json.dumps(experiment_lock, indent=2)
        )
        (version_path / "evaluation.json").write_text(json.dumps(evaluation, indent=2))
        checksum_lines = []
        for name in REQUIRED_FILES:
            if name == "checksums.sha256":
                continue
            digest = _sha256_file(version_path / name)
            checksum_lines.append(f"{digest}  {name}")
        (version_path / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n")
        if promote:
            self.promote_version(version)
        return version_path

    def promote_version(self, version: str) -> None:
        if not self.version_exists(version):
            raise ValueError(
                f"Cannot promote incomplete ppo_discovery version {version}"
            )
        self.verify_checksums(version)
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

    def verify_checksums(self, version: str) -> None:
        version_path = self._version_path(version)
        checksum_file = version_path / "checksums.sha256"
        if not checksum_file.is_file():
            raise ValueError(f"missing checksums.sha256 for {version}")
        expected: dict[str, str] = {}
        for line in checksum_file.read_text().splitlines():
            if not line.strip():
                continue
            digest, name = line.split("  ", 1)
            expected[name] = digest
        for name in REQUIRED_FILES:
            if name == "checksums.sha256":
                continue
            actual = _sha256_file(version_path / name)
            if name not in expected or expected[name] != actual:
                raise ValueError(f"checksum mismatch for {name} in {version}")

    def load_artifacts(self, version: str) -> PPODiscoveryArtifacts:
        version_path = self._version_path(version)
        if not self.version_exists(version):
            raise FileNotFoundError(f"incomplete ppo_discovery version {version}")
        self.verify_checksums(version)
        config = PPODiscoveryConfig.from_dict(
            json.loads((version_path / "config.json").read_text())
        )
        return PPODiscoveryArtifacts(
            policy_state_dict=torch.load(
                version_path / "policy.pt", map_location="cpu", weights_only=True
            ),
            config=config,
            feature_scalers=json.loads(
                (version_path / "feature_scalers.json").read_text()
            ),
            regime_hmm=json.loads((version_path / "regime_hmm.json").read_text()),
            metadata=json.loads((version_path / "metadata.json").read_text()),
            universe_manifest=json.loads(
                (version_path / "universe_manifest.json").read_text()
            ),
            version=version,
            artifact_dir=version_path,
        )

    def load_current_artifacts(self) -> PPODiscoveryArtifacts:
        version = self.read_current_version()
        if version is None:
            raise FileNotFoundError("no promoted ppo_discovery artifact")
        return self.load_artifacts(version)


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "REQUIRED_FILES",
    "PPODiscoveryArtifacts",
    "PPODiscoveryHalalNewModelStorage",
]

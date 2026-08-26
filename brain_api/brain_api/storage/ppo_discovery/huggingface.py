"""HuggingFace Hub storage for ppo_discovery artifacts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from brain_api.core.config import (
    get_hf_ppo_discovery_halal_new_model_repo,
    get_hf_token,
)
from brain_api.storage.ppo_discovery.local import (
    REQUIRED_FILES,
    PPODiscoveryArtifacts,
    PPODiscoveryHalalNewModelStorage,
)


class PPODiscoveryHuggingFaceModelStorage:
    """HF storage for the ``ppo_discovery_halal_new`` bucket."""

    def __init__(
        self,
        *,
        repo_id: str | None = None,
        token: str | None = None,
        local_cache: PPODiscoveryHalalNewModelStorage,
    ) -> None:
        self.repo_id = repo_id or get_hf_ppo_discovery_halal_new_model_repo()
        self.token = token or get_hf_token()
        self.local_cache = local_cache
        self.api = HfApi(token=self.token)

    def download_model(
        self, version: str | None = None, use_cache: bool = True
    ) -> PPODiscoveryArtifacts:
        if self.repo_id is None:
            raise ValueError("HF_PPO_DISCOVERY_HALAL_NEW_MODEL_REPO is not configured")
        revision = version or "main"
        if use_cache and version and self.local_cache.version_exists(version):
            return self.local_cache.load_artifacts(version)
        local_dir = Path(
            snapshot_download(
                repo_id=self.repo_id,
                revision=revision,
                repo_type="model",
                token=self.token,
            )
        )
        resolved_version = json.loads((local_dir / "metadata.json").read_text())[
            "version"
        ]
        dest = self.local_cache._version_path(resolved_version)
        dest.mkdir(parents=True, exist_ok=True)
        for name in REQUIRED_FILES:
            source = local_dir / name
            if source.is_file():
                shutil.copy2(source, dest / name)
        return self.local_cache.load_artifacts(resolved_version)

    def upload_model(self, version: str, *, make_current: bool = False) -> None:
        """Upload a local candidate. ``make_current`` writes HF ``main`` only on promote."""
        if not self.repo_id:
            raise ValueError("HF_PPO_DISCOVERY_HALAL_NEW_MODEL_REPO is not configured")
        local_dir = self.local_cache._version_path(version)
        if not self.local_cache.version_exists(version):
            raise FileNotFoundError(f"incomplete ppo_discovery version {version}")
        self.api.upload_folder(
            folder_path=str(local_dir),
            repo_id=self.repo_id,
            repo_type="model",
            revision=version,
            commit_message=f"ppo_discovery candidate {version}",
        )
        if make_current:
            self.api.upload_folder(
                folder_path=str(local_dir),
                repo_id=self.repo_id,
                repo_type="model",
                revision="main",
                commit_message=f"ppo_discovery promote {version}",
            )


def maybe_upload_ppo_discovery(
    storage: PPODiscoveryHalalNewModelStorage,
    version: str,
    *,
    make_current: bool,
) -> None:
    """Upload when the HF repo env is configured; skip when it is blank."""
    repo = get_hf_ppo_discovery_halal_new_model_repo()
    if not repo:
        return
    PPODiscoveryHuggingFaceModelStorage(repo_id=repo, local_cache=storage).upload_model(
        version, make_current=make_current
    )


__all__ = ["PPODiscoveryHuggingFaceModelStorage", "maybe_upload_ppo_discovery"]

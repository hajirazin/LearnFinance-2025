"""Base HuggingFace Hub storage for model artifacts.

This module provides a generic base class for HuggingFace model storage
that can be subclassed for specific model types (LSTM, PatchTST, etc.).
"""

import json
import logging
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

import joblib
import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError

from brain_api.core.config import get_hf_model_repo, get_hf_token

logger = logging.getLogger(__name__)

# Type variables for generic storage
ConfigT = TypeVar("ConfigT")
ModelT = TypeVar("ModelT")
ArtifactsT = TypeVar("ArtifactsT")
LocalStorageT = TypeVar("LocalStorageT")


@dataclass
class HFModelInfo:
    """Information about a model stored on HuggingFace Hub."""

    repo_id: str
    version: str
    revision: str  # Git commit/tag on HF


class BaseHuggingFaceModelStorage(ABC, Generic[ConfigT, ModelT, ArtifactsT, LocalStorageT]):
    """Base HuggingFace Hub storage for model artifacts.

    Stores model artifacts as files in a HuggingFace Model repository:
        - weights.pt            (PyTorch model weights)
        - feature_scaler.pkl    (sklearn StandardScaler for input features)
        - config.json           (model hyperparameters)
        - metadata.json         (training info, metrics, data window)

    Versions are managed as git tags/branches on the HF repo.
    The 'main' branch typically points to the current promoted version.

    Subclasses must implement:
        - model_type: str property (e.g., "lstm", "patchtst")
        - _create_local_storage(): Create local storage instance
        - _load_config(config_dict): Load config from dict
        - _create_model(config): Create model instance
        - _generate_readme(version, metadata): Generate README content
    """

    def __init__(
        self,
        repo_id: str | None = None,
        token: str | None = None,
        local_cache: LocalStorageT | None = None,
    ):
        """Initialize HuggingFace model storage.

        Args:
            repo_id: HuggingFace repo ID (e.g., 'username/learnfinance-model').
                     Defaults to HF_MODEL_REPO env var.
            token: HuggingFace API token. If None, uses HF_TOKEN env var or
                   cached token from `huggingface-cli login`.
            local_cache: Optional local storage for caching downloaded models.
        """
        self.repo_id = repo_id or get_hf_model_repo()
        self.token = token or get_hf_token()
        self.local_cache = local_cache or self._create_local_storage()
        self.api = HfApi(token=self.token)

        if not self.repo_id:
            raise ValueError(
                "HuggingFace model repo not configured. "
                "Set HF_MODEL_REPO environment variable or pass repo_id."
            )

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return model type identifier (e.g., 'lstm', 'patchtst')."""
        pass

    @abstractmethod
    def _create_local_storage(self) -> LocalStorageT:
        """Create local storage instance for caching."""
        pass

    @abstractmethod
    def _load_config(self, config_dict: dict[str, Any]) -> ConfigT:
        """Load config from dictionary."""
        pass

    @abstractmethod
    def _create_model(self, config: ConfigT) -> ModelT:
        """Create model instance from config."""
        pass

    @abstractmethod
    def _create_artifacts(
        self,
        config: ConfigT,
        feature_scaler: Any,
        model: ModelT,
        version: str,
    ) -> ArtifactsT:
        """Create artifacts instance."""
        pass

    @abstractmethod
    def _generate_readme(self, version: str, metadata: dict[str, Any]) -> str:
        """Generate README content for the model card."""
        pass

    def _ensure_repo_exists(self) -> None:
        """Create the HF repo if it doesn't exist."""
        try:
            self.api.repo_info(repo_id=self.repo_id, repo_type="model")
        except RepositoryNotFoundError:
            logger.info(f"Creating HuggingFace model repo: {self.repo_id}")
            self.api.create_repo(
                repo_id=self.repo_id,
                repo_type="model",
                exist_ok=True,
            )

    def upload_model(
        self,
        version: str,
        model: ModelT,
        feature_scaler: Any,
        config: ConfigT,
        metadata: dict[str, Any],
        make_current: bool = False,
    ) -> HFModelInfo:
        """Upload model artifacts to HuggingFace Hub.

        Args:
            version: Version string (e.g., 'v2025-01-05-abc123')
            model: Trained PyTorch model
            feature_scaler: Fitted StandardScaler for input features
            config: Model configuration
            metadata: Training metadata
            make_current: If True, also update 'main' branch to point to this version

        Returns:
            HFModelInfo with upload details
        """
        self._ensure_repo_exists()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Save model weights
            weights_path = tmppath / "weights.pt"
            torch.save(model.state_dict(), weights_path)

            # Save feature scaler
            scaler_path = tmppath / "feature_scaler.pkl"
            joblib.dump(feature_scaler, scaler_path)

            # Save config
            config_path = tmppath / "config.json"
            with open(config_path, "w") as f:
                json.dump(config.to_dict(), f, indent=2)

            # Save metadata
            metadata_path = tmppath / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

            # Create README
            readme_path = tmppath / "README.md"
            readme_content = self._generate_readme(version, metadata)
            with open(readme_path, "w") as f:
                f.write(readme_content)

            # Upload all files to the version branch/tag
            logger.info(f"Uploading {self.model_type.upper()} model {version} to {self.repo_id}")

            self.api.upload_folder(
                folder_path=tmpdir,
                repo_id=self.repo_id,
                repo_type="model",
                revision=version,
                commit_message=f"Add {self.model_type.upper()} model version {version}",
            )

            if make_current:
                logger.info(f"Setting {version} as current (main branch)")
                self.api.upload_folder(
                    folder_path=tmpdir,
                    repo_id=self.repo_id,
                    repo_type="model",
                    revision="main",
                    commit_message=f"Promote {self.model_type.upper()} version {version} to current",
                )

        return HFModelInfo(
            repo_id=self.repo_id,
            version=version,
            revision=version,
        )

    def download_model(
        self,
        version: str | None = None,
        use_cache: bool = True,
    ) -> ArtifactsT:
        """Download model artifacts from HuggingFace Hub.

        Args:
            version: Version string to download. If None, downloads 'main' (current).
            use_cache: If True, check local cache first and update it after download.

        Returns:
            Model artifacts ready for inference
        """
        revision = version or "main"

        # Check local cache first
        if use_cache and version:
            if self.local_cache.version_exists(version):
                logger.info(f"Loading {self.model_type.upper()} model {version} from local cache")
                return self.local_cache.load_current_artifacts()

        logger.info(f"Downloading {self.model_type.upper()} model from {self.repo_id} (revision: {revision})")

        # Download all files to a temp directory
        local_dir = snapshot_download(
            repo_id=self.repo_id,
            revision=revision,
            repo_type="model",
            token=self.token,
        )
        local_path = Path(local_dir)

        # Load config
        config_path = local_path / "config.json"
        with open(config_path) as f:
            config_dict = json.load(f)
        config = self._load_config(config_dict)

        # Load feature scaler
        scaler_path = local_path / "feature_scaler.pkl"
        feature_scaler = joblib.load(scaler_path)

        # Load model
        weights_path = local_path / "weights.pt"
        model = self._create_model(config)
        model.load_state_dict(torch.load(weights_path, weights_only=True))
        model.eval()

        # Determine actual version from metadata if we downloaded 'main'
        actual_version = version
        if not version:
            metadata_path = local_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                    actual_version = metadata.get("version", "main")

        # Cache locally if we have a version
        if use_cache and actual_version and actual_version != "main":
            logger.info(f"Caching {self.model_type.upper()} model {actual_version} locally")
            metadata_path = local_path / "metadata.json"
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)

            self.local_cache.write_artifacts(
                version=actual_version,
                model=model,
                feature_scaler=feature_scaler,
                config=config,
                metadata=metadata,
            )
            self.local_cache.promote_version(actual_version)

        return self._create_artifacts(
            config=config,
            feature_scaler=feature_scaler,
            model=model,
            version=actual_version or "main",
        )

    def get_current_version(self) -> str | None:
        """Get the current (main branch) version from HF Hub."""
        try:
            metadata_path = hf_hub_download(
                repo_id=self.repo_id,
                filename="metadata.json",
                repo_type="model",
                token=self.token,
            )
            with open(metadata_path) as f:
                metadata = json.load(f)
            return metadata.get("version")
        except Exception:
            return None

    def list_versions(self) -> list[str]:
        """List all available model versions (branches) on HF Hub."""
        try:
            refs = self.api.list_repo_refs(repo_id=self.repo_id, repo_type="model")
            versions = [
                branch.name
                for branch in refs.branches
                if branch.name.startswith("v")
            ]
            return sorted(versions, reverse=True)
        except Exception:
            return []




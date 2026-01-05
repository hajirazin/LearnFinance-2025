"""HuggingFace Hub storage for LSTM model artifacts."""

import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError

from brain_api.core.config import get_hf_model_repo, get_hf_token
from brain_api.core.lstm import LSTMConfig, LSTMModel
from brain_api.storage.lstm.local import LSTMArtifacts, LocalModelStorage

logger = logging.getLogger(__name__)


@dataclass
class HFModelInfo:
    """Information about a model stored on HuggingFace Hub."""

    repo_id: str
    version: str
    revision: str  # Git commit/tag on HF


class HuggingFaceModelStorage:
    """HuggingFace Hub storage for LSTM model artifacts.

    Stores model artifacts as files in a HuggingFace Model repository:
        - weights.pt            (PyTorch model weights)
        - feature_scaler.pkl    (sklearn StandardScaler for input features)
        - config.json           (model hyperparameters)
        - metadata.json         (training info, metrics, data window)

    Versions are managed as git tags/branches on the HF repo.
    The 'main' branch typically points to the current promoted version.

    Authentication:
        The huggingface_hub library automatically uses credentials from:
        1. Explicit token parameter (if provided)
        2. HF_TOKEN environment variable
        3. Cached token from `huggingface-cli login` (~/.cache/huggingface/token)

        Recommended: Run `huggingface-cli login` once, then no token needed.

    Note: This class also maintains a local cache to avoid re-downloading
    models on every inference request.
    """

    def __init__(
        self,
        repo_id: str | None = None,
        token: str | None = None,
        local_cache: LocalModelStorage | None = None,
    ):
        """Initialize HuggingFace model storage.

        Args:
            repo_id: HuggingFace repo ID (e.g., 'username/learnfinance-lstm').
                     Defaults to HF_MODEL_REPO env var.
            token: HuggingFace API token. If None, uses HF_TOKEN env var or
                   cached token from `huggingface-cli login`.
            local_cache: Optional LocalModelStorage for caching downloaded models.
        """
        self.repo_id = repo_id or get_hf_model_repo()
        self.token = token or get_hf_token()  # None is OK - HfApi uses cached token
        self.local_cache = local_cache or LocalModelStorage()
        self.api = HfApi(token=self.token)

        if not self.repo_id:
            raise ValueError(
                "HuggingFace model repo not configured. "
                "Set HF_MODEL_REPO environment variable or pass repo_id."
            )

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
        model: LSTMModel,
        feature_scaler: Any,
        config: LSTMConfig,
        metadata: dict[str, Any],
        make_current: bool = False,
    ) -> HFModelInfo:
        """Upload model artifacts to HuggingFace Hub.

        Args:
            version: Version string (e.g., 'v2025-01-05-abc123')
            model: Trained PyTorch LSTM model
            feature_scaler: Fitted StandardScaler for input features
            config: Model configuration
            metadata: Training metadata
            make_current: If True, also update 'main' branch to point to this version

        Returns:
            HFModelInfo with upload details
        """
        self._ensure_repo_exists()

        # Create a temporary directory with all artifacts
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

            # Create a simple README for the model card
            readme_path = tmppath / "README.md"
            readme_content = f"""---
tags:
- lstm
- finance
- weekly-returns
- learnfinance
---

# LearnFinance LSTM Model - {version}

LSTM model for predicting weekly stock returns.

## Model Details

- **Version**: {version}
- **Training Window**: {metadata.get('data_window', {}).get('start', 'N/A')} to {metadata.get('data_window', {}).get('end', 'N/A')}
- **Symbols**: {len(metadata.get('symbols', []))} stocks

## Metrics

- Train Loss: {metadata.get('metrics', {}).get('train_loss', 'N/A')}
- Validation Loss: {metadata.get('metrics', {}).get('val_loss', 'N/A')}
- Baseline Loss: {metadata.get('metrics', {}).get('baseline_loss', 'N/A')}

## Usage

```python
from brain_api.storage.lstm import HuggingFaceModelStorage

storage = HuggingFaceModelStorage(repo_id="{self.repo_id}")
artifacts = storage.load_model(version="{version}")
```
"""
            with open(readme_path, "w") as f:
                f.write(readme_content)

            # Upload all files to the version branch/tag
            logger.info(f"Uploading model {version} to {self.repo_id}")

            # Upload to a branch named after the version
            self.api.upload_folder(
                folder_path=tmpdir,
                repo_id=self.repo_id,
                repo_type="model",
                revision=version,
                commit_message=f"Add model version {version}",
            )

            # If make_current, also update main branch
            if make_current:
                logger.info(f"Setting {version} as current (main branch)")
                self.api.upload_folder(
                    folder_path=tmpdir,
                    repo_id=self.repo_id,
                    repo_type="model",
                    revision="main",
                    commit_message=f"Promote version {version} to current",
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
    ) -> LSTMArtifacts:
        """Download model artifacts from HuggingFace Hub.

        Args:
            version: Version string to download. If None, downloads 'main' (current).
            use_cache: If True, check local cache first and update it after download.

        Returns:
            LSTMArtifacts with loaded model, scaler, and config
        """
        revision = version or "main"

        # Check local cache first
        if use_cache and version:
            if self.local_cache.version_exists(version):
                logger.info(f"Loading model {version} from local cache")
                return self.local_cache.load_current_artifacts()

        logger.info(f"Downloading model from {self.repo_id} (revision: {revision})")

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
        config = LSTMConfig(**config_dict)

        # Load feature scaler
        scaler_path = local_path / "feature_scaler.pkl"
        feature_scaler = joblib.load(scaler_path)

        # Load model
        weights_path = local_path / "weights.pt"
        model = LSTMModel(config)
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
            logger.info(f"Caching model {actual_version} locally")
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

        return LSTMArtifacts(
            config=config,
            feature_scaler=feature_scaler,
            model=model,
            version=actual_version or "main",
        )

    def get_current_version(self) -> str | None:
        """Get the current (main branch) version from HF Hub.

        Returns:
            Version string if metadata exists, None otherwise.
        """
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
        """List all available model versions (branches) on HF Hub.

        Returns:
            List of version strings
        """
        try:
            refs = self.api.list_repo_refs(repo_id=self.repo_id, repo_type="model")
            # Return branch names that look like versions (start with 'v')
            versions = [
                branch.name
                for branch in refs.branches
                if branch.name.startswith("v")
            ]
            return sorted(versions, reverse=True)
        except Exception:
            return []


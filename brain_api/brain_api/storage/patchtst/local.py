"""Local filesystem storage for PatchTST model artifacts."""

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import torch
from sklearn.preprocessing import StandardScaler

from brain_api.storage.base import DEFAULT_DATA_PATH

if TYPE_CHECKING:
    from transformers import PatchTSTForPrediction

    from brain_api.core.patchtst import PatchTSTConfig


@dataclass
class PatchTSTArtifacts:
    """Loaded PatchTST model artifacts for inference.

    Contains everything needed to run inference:
    - config: model hyperparameters
    - feature_scaler: fitted StandardScaler for input normalization
    - model: HuggingFace PatchTSTForPrediction model with loaded weights
    - version: the version string these artifacts came from
    """

    config: "PatchTSTConfig"
    feature_scaler: StandardScaler
    model: "PatchTSTForPrediction"
    version: str


class PatchTSTModelStorage:
    """Local filesystem storage for PatchTST model artifacts.

    Artifacts are stored under:
        {base_path}/models/patchtst/{version}/
            - weights.pt            (PyTorch model weights)
            - feature_scaler.pkl    (sklearn StandardScaler for input features)
            - config.json           (model hyperparameters)
            - metadata.json         (training info, metrics, data window)

    The current version pointer is stored at:
        {base_path}/models/patchtst/current
    """

    def __init__(self, base_path: Path | str | None = None):
        """Initialize storage.

        Args:
            base_path: Base path for data storage. Defaults to 'data/'.
        """
        if base_path is None:
            base_path = DEFAULT_DATA_PATH
        self.base_path = Path(base_path)
        self.patchtst_path = self.base_path / "models" / "patchtst"

    def _version_path(self, version: str) -> Path:
        """Get the path for a specific version."""
        return self.patchtst_path / version

    def version_exists(self, version: str) -> bool:
        """Check if a version already exists."""
        return self._version_path(version).exists()

    def write_artifacts(
        self,
        version: str,
        model: "PatchTSTForPrediction",
        feature_scaler: StandardScaler,
        config: "PatchTSTConfig",
        metadata: dict[str, Any],
    ) -> Path:
        """Write model artifacts for a version.

        Args:
            version: Version string (e.g., 'v2025-01-05-abc123')
            model: Trained HuggingFace PatchTSTForPrediction model
            feature_scaler: Fitted StandardScaler for input features
            config: Model configuration
            metadata: Training metadata (includes window, metrics, etc.)

        Returns:
            Path to the version directory
        """
        version_dir = self._version_path(version)
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights (PyTorch state dict)
        weights_path = version_dir / "weights.pt"
        torch.save(model.state_dict(), weights_path)

        # Save feature scaler
        feature_scaler_path = version_dir / "feature_scaler.pkl"
        joblib.dump(feature_scaler, feature_scaler_path)

        # Save config
        config_path = version_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config.to_dict(), f, indent=2)

        # Save metadata
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        return version_dir

    def read_current_version(self) -> str | None:
        """Read the current version pointer.

        Returns:
            Version string if current exists, None otherwise.
        """
        current_file = self.patchtst_path / "current"
        if not current_file.exists():
            return None
        return current_file.read_text().strip()

    def read_metadata(self, version: str) -> dict[str, Any] | None:
        """Read metadata for a version.

        Args:
            version: Version string

        Returns:
            Metadata dict if exists, None otherwise.
        """
        metadata_path = self._version_path(version) / "metadata.json"
        if not metadata_path.exists():
            return None
        with open(metadata_path) as f:
            return json.load(f)

    def promote_version(self, version: str) -> None:
        """Atomically promote a version to current.

        Args:
            version: Version string to promote
        """
        self.patchtst_path.mkdir(parents=True, exist_ok=True)
        current_file = self.patchtst_path / "current"

        fd, temp_path = tempfile.mkstemp(
            dir=self.patchtst_path,
            prefix=".current_",
            suffix=".tmp",
        )
        try:
            os.write(fd, version.encode())
            os.close(fd)
            os.rename(temp_path, current_file)
        except Exception:
            os.close(fd) if fd else None
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    # ========================================================================
    # Inference artifact loading
    # ========================================================================

    def load_config(self, version: str) -> "PatchTSTConfig":
        """Load model configuration for a version.

        Args:
            version: Version string

        Returns:
            PatchTSTConfig reconstructed from config.json

        Raises:
            FileNotFoundError: if config.json doesn't exist
        """
        from brain_api.core.patchtst import PatchTSTConfig

        config_path = self._version_path(version) / "config.json"
        with open(config_path) as f:
            config_dict = json.load(f)
        return PatchTSTConfig(**config_dict)

    def load_feature_scaler(self, version: str) -> StandardScaler:
        """Load fitted feature scaler for a version.

        Args:
            version: Version string

        Returns:
            StandardScaler loaded from feature_scaler.pkl

        Raises:
            FileNotFoundError: if feature_scaler.pkl doesn't exist
        """
        scaler_path = self._version_path(version) / "feature_scaler.pkl"
        return joblib.load(scaler_path)

    def load_model(
        self, version: str, config: "PatchTSTConfig | None" = None
    ) -> "PatchTSTForPrediction":
        """Load trained PatchTST model for a version.

        Args:
            version: Version string
            config: Optional PatchTSTConfig; if not provided, will load from config.json

        Returns:
            PatchTSTForPrediction with weights loaded

        Raises:
            FileNotFoundError: if weights.pt doesn't exist
        """
        from transformers import PatchTSTForPrediction

        if config is None:
            config = self.load_config(version)

        weights_path = self._version_path(version) / "weights.pt"
        hf_config = config.to_hf_config()
        model = PatchTSTForPrediction(hf_config)
        model.load_state_dict(torch.load(weights_path, weights_only=True))
        model.eval()
        return model

    def load_current_artifacts(self) -> PatchTSTArtifacts:
        """Load all artifacts for the current promoted version.

        Returns:
            PatchTSTArtifacts containing everything needed for inference

        Raises:
            ValueError: if no current version is set
            FileNotFoundError: if any artifact file is missing
        """
        version = self.read_current_version()
        if version is None:
            raise ValueError(
                "No current PatchTST version set. Train a model first with POST /train/patchtst"
            )

        config = self.load_config(version)
        feature_scaler = self.load_feature_scaler(version)
        model = self.load_model(version, config)

        return PatchTSTArtifacts(
            config=config,
            feature_scaler=feature_scaler,
            model=model,
            version=version,
        )


def create_metadata(
    version: str,
    data_window_start: str,
    data_window_end: str,
    symbols: list[str],
    config: "PatchTSTConfig",
    train_loss: float,
    val_loss: float,
    baseline_loss: float,
    promoted: bool,
    prior_version: str | None,
) -> dict[str, Any]:
    """Create metadata dict for a PatchTST training run.

    Args:
        version: Version string
        data_window_start: Training data start date (ISO format)
        data_window_end: Training data end date (ISO format)
        symbols: List of symbols used for training
        config: PatchTST model configuration
        train_loss: Final training loss
        val_loss: Validation loss
        baseline_loss: Baseline (persistence) loss
        promoted: Whether this version was promoted to current
        prior_version: Previous current version (if any)

    Returns:
        Metadata dictionary
    """
    return {
        "model_type": "patchtst",
        "version": version,
        "training_timestamp": datetime.now(UTC).isoformat(),
        "data_window": {
            "start": data_window_start,
            "end": data_window_end,
        },
        "symbols": symbols,
        "config": config.to_dict(),
        "metrics": {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "baseline_loss": baseline_loss,
        },
        "promoted": promoted,
        "prior_version": prior_version,
    }


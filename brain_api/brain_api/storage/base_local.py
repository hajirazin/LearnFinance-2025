"""Base local filesystem storage for model artifacts.

This module provides a generic base class for local model storage
that can be subclassed for specific model types (LSTM, PatchTST, etc.).
"""

import json
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, TypeVar

import joblib
import torch
from sklearn.preprocessing import StandardScaler

from brain_api.storage.base import DEFAULT_DATA_PATH

# Type variables for generic storage
ConfigT = TypeVar("ConfigT")
ModelT = TypeVar("ModelT")
ArtifactsT = TypeVar("ArtifactsT")


class BaseLocalModelStorage(ABC, Generic[ConfigT, ModelT, ArtifactsT]):
    """Base local filesystem storage for model artifacts.

    Artifacts are stored under:
        {base_path}/models/{model_type}/{version}/
            - weights.pt            (PyTorch model weights)
            - feature_scaler.pkl    (sklearn StandardScaler for input features)
            - config.json           (model hyperparameters)
            - metadata.json         (training info, metrics, data window)

    The current version pointer is stored at:
        {base_path}/models/{model_type}/current

    Subclasses must implement:
        - model_type: str property (e.g., "lstm", "patchtst")
        - _load_config(config_dict): Load config from dict
        - _create_model(config): Create model instance from config
        - _create_artifacts(...): Create artifacts dataclass instance
    """

    def __init__(self, base_path: Path | str | None = None):
        """Initialize storage.

        Args:
            base_path: Base path for data storage. Defaults to 'data/'.
        """
        if base_path is None:
            base_path = DEFAULT_DATA_PATH
        self.base_path = Path(base_path)
        self._model_path = self.base_path / "models" / self.model_type

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return model type identifier (e.g., 'lstm', 'patchtst')."""
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
        feature_scaler: StandardScaler,
        model: ModelT,
        version: str,
    ) -> ArtifactsT:
        """Create artifacts dataclass instance."""
        pass

    def _version_path(self, version: str) -> Path:
        """Get the path for a specific version."""
        return self._model_path / version

    def version_exists(self, version: str) -> bool:
        """Check if a version already exists."""
        return self._version_path(version).exists()

    def write_artifacts(
        self,
        version: str,
        model: ModelT,
        feature_scaler: StandardScaler,
        config: ConfigT,
        metadata: dict[str, Any],
    ) -> Path:
        """Write model artifacts for a version.

        Args:
            version: Version string (e.g., 'v2025-01-05-abc123')
            model: Trained PyTorch model
            feature_scaler: Fitted StandardScaler for input features
            config: Model configuration
            metadata: Training metadata (includes window, metrics, etc.)

        Returns:
            Path to the version directory
        """
        version_dir = self._version_path(version)
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights
        weights_path = version_dir / "weights.pt"
        torch.save(model.state_dict(), weights_path)

        # Save feature scaler (for input normalization)
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
        current_file = self._model_path / "current"
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

        Uses atomic write (write to temp file, then rename) to ensure
        the current pointer is never in a partial state.

        Args:
            version: Version string to promote
        """
        # Ensure model directory exists
        self._model_path.mkdir(parents=True, exist_ok=True)

        current_file = self._model_path / "current"

        # Atomic write: write to temp file in same directory, then rename
        fd, temp_path = tempfile.mkstemp(
            dir=self._model_path,
            prefix=".current_",
            suffix=".tmp",
        )
        try:
            os.write(fd, version.encode())
            os.close(fd)
            # Atomic rename on POSIX
            os.rename(temp_path, current_file)
        except Exception:
            # Clean up temp file on error
            os.close(fd) if fd else None
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    # ========================================================================
    # Inference artifact loading
    # ========================================================================

    def load_config(self, version: str) -> ConfigT:
        """Load model configuration for a version.

        Args:
            version: Version string

        Returns:
            Config object reconstructed from config.json

        Raises:
            FileNotFoundError: if config.json doesn't exist
        """
        config_path = self._version_path(version) / "config.json"
        with open(config_path) as f:
            config_dict = json.load(f)
        return self._load_config(config_dict)

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

    def load_model(self, version: str, config: ConfigT | None = None) -> ModelT:
        """Load trained model for a version.

        Args:
            version: Version string
            config: Optional config; if not provided, will load from config.json

        Returns:
            Model with weights loaded

        Raises:
            FileNotFoundError: if weights.pt doesn't exist
        """
        if config is None:
            config = self.load_config(version)

        weights_path = self._version_path(version) / "weights.pt"
        model = self._create_model(config)
        model.load_state_dict(torch.load(weights_path, weights_only=True))
        model.eval()  # Set to evaluation mode for inference
        return model

    def load_current_artifacts(self) -> ArtifactsT:
        """Load all artifacts for the current promoted version.

        Convenience method that loads config, scaler, and model together.

        Returns:
            Artifacts containing everything needed for inference

        Raises:
            ValueError: if no current version is set
            FileNotFoundError: if any artifact file is missing
        """
        version = self.read_current_version()
        if version is None:
            raise ValueError(
                f"No current {self.model_type.upper()} version set. "
                f"Train a model first with POST /train/{self.model_type}"
            )

        config = self.load_config(version)
        feature_scaler = self.load_feature_scaler(version)
        model = self.load_model(version, config)

        return self._create_artifacts(
            config=config,
            feature_scaler=feature_scaler,
            model=model,
            version=version,
        )



"""Local filesystem storage for model artifacts."""

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import torch

from brain_api.core.lstm import LSTMConfig, LSTMModel, StandardScaler

# Default base path for data storage
DEFAULT_DATA_PATH = Path("data")


class LocalModelStorage:
    """Local filesystem storage for LSTM model artifacts.

    Artifacts are stored under:
        {base_path}/models/lstm/{version}/
            - weights.pt            (PyTorch model weights)
            - feature_scaler.pkl    (sklearn StandardScaler for input features)
            - config.json           (model hyperparameters)
            - metadata.json         (training info, metrics, data window)

    The current version pointer is stored at:
        {base_path}/models/lstm/current

    Note: The model predicts weekly returns directly, so no price_scaler
    is needed for denormalization.
    """

    def __init__(self, base_path: Path | str | None = None):
        """Initialize storage.

        Args:
            base_path: Base path for data storage. Defaults to 'data/'.
        """
        if base_path is None:
            base_path = DEFAULT_DATA_PATH
        self.base_path = Path(base_path)
        self.lstm_path = self.base_path / "models" / "lstm"

    def _version_path(self, version: str) -> Path:
        """Get the path for a specific version."""
        return self.lstm_path / version

    def version_exists(self, version: str) -> bool:
        """Check if a version already exists."""
        return self._version_path(version).exists()

    def write_artifacts(
        self,
        version: str,
        model: LSTMModel,
        feature_scaler: StandardScaler,
        config: LSTMConfig,
        metadata: dict[str, Any],
    ) -> Path:
        """Write model artifacts for a version.

        Args:
            version: Version string (e.g., 'v2025-01-05-abc123')
            model: Trained PyTorch LSTM model
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
        current_file = self.lstm_path / "current"
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
        # Ensure lstm directory exists
        self.lstm_path.mkdir(parents=True, exist_ok=True)

        current_file = self.lstm_path / "current"

        # Atomic write: write to temp file in same directory, then rename
        fd, temp_path = tempfile.mkstemp(
            dir=self.lstm_path,
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


def create_metadata(
    version: str,
    data_window_start: str,
    data_window_end: str,
    symbols: list[str],
    config: LSTMConfig,
    train_loss: float,
    val_loss: float,
    baseline_loss: float,
    promoted: bool,
    prior_version: str | None,
) -> dict[str, Any]:
    """Create metadata dict for a training run.

    Args:
        version: Version string
        data_window_start: Training data start date (ISO format)
        data_window_end: Training data end date (ISO format)
        symbols: List of symbols used for training
        config: Model configuration
        train_loss: Final training loss
        val_loss: Validation loss
        baseline_loss: Baseline (persistence) loss
        promoted: Whether this version was promoted to current
        prior_version: Previous current version (if any)

    Returns:
        Metadata dictionary
    """
    return {
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

"""Local filesystem storage for forecaster snapshots.

Stores yearly LSTM and PatchTST model snapshots for walk-forward
forecast generation during RL training.

Also supports HuggingFace Hub upload/download using the same repo
as the main model but with different branch naming convention:
- Main model: v2025-01-05-abc123
- Snapshots: snapshot-2024-12-31
"""

import json
import logging
import pickle
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

import torch
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError
from sklearn.preprocessing import StandardScaler

from brain_api.core.config import (
    get_hf_lstm_model_repo,
    get_hf_patchtst_model_repo,
    get_hf_token,
)
from brain_api.storage.base import DEFAULT_DATA_PATH

logger = logging.getLogger(__name__)


@dataclass
class LSTMSnapshotArtifacts:
    """Loaded LSTM snapshot artifacts for inference.

    Contains everything needed to run inference:
    - config: model hyperparameters
    - feature_scaler: fitted StandardScaler for input normalization
    - model: PyTorch LSTM model with loaded weights
    - cutoff_date: the data cutoff date for this snapshot
    """

    config: Any  # LSTMConfig
    feature_scaler: StandardScaler
    model: Any  # LSTMModel
    cutoff_date: date


@dataclass
class PatchTSTSnapshotArtifacts:
    """Loaded PatchTST snapshot artifacts for inference.

    Contains everything needed to run inference:
    - config: model hyperparameters
    - feature_scaler: fitted StandardScaler for input normalization
    - model: HuggingFace PatchTSTForPrediction model
    - cutoff_date: the data cutoff date for this snapshot
    """

    config: Any  # PatchTSTConfig
    feature_scaler: StandardScaler
    model: Any  # PatchTSTForPrediction
    cutoff_date: date


class SnapshotLocalStorage:
    """Local filesystem storage for forecaster snapshots.

    Snapshots are stored as siblings to main model versions with flat structure:
        {base_path}/models/{forecaster_type}/snapshot-{cutoff_date}/
            - weights.pt            (PyTorch model weights)
            - feature_scaler.pkl    (sklearn StandardScaler)
            - config.json           (model hyperparameters)
            - metadata.json         (training info)

    Example:
        data/models/lstm/v2024-01-01-abc123/    # Main model version
        data/models/lstm/snapshot-2019-12-31/   # Snapshot (same flat structure)
        data/models/lstm/snapshot-2020-12-31/   # Snapshot
        data/models/patchtst/snapshot-2019-12-31/

    Pattern-based identification:
        - Main model: v{date}-{hash}
        - Snapshot: snapshot-{date}

    HuggingFace Support:
        Snapshots use the same repo as the main model with branch naming:
        - Main model versions: v2025-01-05-abc123
        - Snapshot branches: snapshot-2024-12-31
    """

    def __init__(
        self,
        forecaster_type: Literal["lstm", "patchtst"],
        base_path: Path | str | None = None,
        hf_token: str | None = None,
    ):
        """Initialize storage.

        Args:
            forecaster_type: "lstm" or "patchtst"
            base_path: Base path for data storage. Defaults to 'data/'.
            hf_token: HuggingFace API token. If None, uses HF_TOKEN env var.
        """
        if base_path is None:
            base_path = DEFAULT_DATA_PATH
        self.base_path = Path(base_path)
        self.forecaster_type = forecaster_type
        self._hf_token = hf_token
        # Models directory where both main versions and snapshots live as siblings
        self._models_path = self.base_path / "models" / forecaster_type

    def _get_hf_repo(self) -> str | None:
        """Get the HuggingFace repo ID for this forecaster type."""
        if self.forecaster_type == "lstm":
            return get_hf_lstm_model_repo()
        else:
            return get_hf_patchtst_model_repo()

    def _get_hf_token(self) -> str | None:
        """Get HF token from instance or environment."""
        return self._hf_token or get_hf_token()

    def _snapshot_branch_name(self, cutoff_date: date) -> str:
        """Get the HF branch name for a snapshot."""
        return f"snapshot-{cutoff_date.isoformat()}"

    def _snapshot_path(self, cutoff_date: date) -> Path:
        """Get the path for a specific snapshot (sibling to main versions)."""
        return self._models_path / f"snapshot-{cutoff_date.isoformat()}"

    def snapshot_exists(self, cutoff_date: date) -> bool:
        """Check if a snapshot already exists locally."""
        return self._snapshot_path(cutoff_date).exists()

    def snapshot_exists_anywhere(
        self, cutoff_date: date, check_hf: bool = False
    ) -> bool:
        """Check if snapshot exists locally OR on HuggingFace (if check_hf=True).

        This is useful when deciding whether to create a new snapshot during training.
        When STORAGE_BACKEND=hf, we should check HF to avoid redundant training.

        Args:
            cutoff_date: The snapshot cutoff date to check
            check_hf: If True, also check HuggingFace for the snapshot

        Returns:
            True if snapshot exists locally or on HF (when check_hf=True)
        """
        if self.snapshot_exists(cutoff_date):
            return True

        if check_hf:
            hf_snapshots = self.list_hf_snapshots()
            return cutoff_date in hf_snapshots

        return False

    def list_snapshots(self) -> list[date]:
        """List all available snapshot cutoff dates.

        Returns:
            Sorted list of cutoff dates.
        """
        if not self._models_path.exists():
            return []

        cutoff_dates = []
        for entry in self._models_path.iterdir():
            if entry.is_dir() and entry.name.startswith("snapshot-"):
                try:
                    date_str = entry.name.replace("snapshot-", "")
                    cutoff_dates.append(date.fromisoformat(date_str))
                except ValueError:
                    continue

        return sorted(cutoff_dates)

    def write_snapshot(
        self,
        cutoff_date: date,
        model: Any,
        feature_scaler: StandardScaler,
        config: Any,
        metadata: dict[str, Any],
    ) -> Path:
        """Write a snapshot for a specific cutoff date.

        Args:
            cutoff_date: The data cutoff date (e.g., 2019-12-31)
            model: Trained LSTM or PatchTST model
            feature_scaler: Fitted StandardScaler
            config: Model configuration
            metadata: Training metadata

        Returns:
            Path to the snapshot directory
        """
        snapshot_dir = self._snapshot_path(cutoff_date)
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights
        weights_path = snapshot_dir / "weights.pt"
        torch.save(model.state_dict(), weights_path)

        # Save scaler
        scaler_path = snapshot_dir / "feature_scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(feature_scaler, f)

        # Save config
        config_path = snapshot_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config.to_dict(), f, indent=2)

        # Save metadata
        metadata_path = snapshot_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        return snapshot_dir

    def load_snapshot(
        self, cutoff_date: date
    ) -> LSTMSnapshotArtifacts | PatchTSTSnapshotArtifacts:
        """Load a snapshot for a specific cutoff date.

        Args:
            cutoff_date: The data cutoff date

        Returns:
            Loaded snapshot artifacts
        """
        snapshot_dir = self._snapshot_path(cutoff_date)

        if not snapshot_dir.exists():
            raise ValueError(f"Snapshot not found for cutoff date {cutoff_date}")

        # Load config
        config_path = snapshot_dir / "config.json"
        with open(config_path) as f:
            config_dict = json.load(f)

        # Load scaler
        scaler_path = snapshot_dir / "feature_scaler.pkl"
        with open(scaler_path, "rb") as f:
            feature_scaler = pickle.load(f)

        # Load model
        weights_path = snapshot_dir / "weights.pt"

        if self.forecaster_type == "lstm":
            from brain_api.core.lstm import LSTMConfig, LSTMModel

            config = LSTMConfig(**config_dict)
            model = LSTMModel(config)
            model.load_state_dict(torch.load(weights_path, weights_only=True))
            model.eval()

            return LSTMSnapshotArtifacts(
                config=config,
                feature_scaler=feature_scaler,
                model=model,
                cutoff_date=cutoff_date,
            )
        else:
            from transformers import PatchTSTForPrediction

            from brain_api.core.patchtst import PatchTSTConfig

            config = PatchTSTConfig(**config_dict)
            hf_config = config.to_hf_config()
            model = PatchTSTForPrediction(hf_config)
            model.load_state_dict(torch.load(weights_path, weights_only=True))
            model.eval()

            return PatchTSTSnapshotArtifacts(
                config=config,
                feature_scaler=feature_scaler,
                model=model,
                cutoff_date=cutoff_date,
            )

    def read_metadata(self, cutoff_date: date) -> dict[str, Any] | None:
        """Read metadata for a snapshot.

        Args:
            cutoff_date: The data cutoff date

        Returns:
            Metadata dict if exists, None otherwise.
        """
        metadata_path = self._snapshot_path(cutoff_date) / "metadata.json"
        if not metadata_path.exists():
            return None
        with open(metadata_path) as f:
            return json.load(f)

    def get_snapshot_for_year(self, year: int) -> date | None:
        """Get the snapshot cutoff date to use for predictions in a given year.

        For year N, we need a snapshot trained on data up to Dec 31 of year N-1.

        Args:
            year: The year we want predictions for

        Returns:
            Cutoff date of the snapshot to use, or None if not available
        """
        target_cutoff = date(year - 1, 12, 31)

        # First try exact match
        if self.snapshot_exists(target_cutoff):
            return target_cutoff

        # Find the closest available snapshot that's <= target
        available = self.list_snapshots()
        valid_snapshots = [d for d in available if d <= target_cutoff]

        if valid_snapshots:
            return max(valid_snapshots)  # Most recent valid snapshot

        return None

    # =========================================================================
    # HuggingFace Hub Methods
    # =========================================================================

    def _ensure_hf_repo_exists(self, api: HfApi, repo_id: str) -> None:
        """Create the HF repo if it doesn't exist."""
        try:
            api.repo_info(repo_id=repo_id, repo_type="model")
        except RepositoryNotFoundError:
            logger.info(f"Creating HuggingFace model repo: {repo_id}")
            api.create_repo(
                repo_id=repo_id,
                repo_type="model",
                exist_ok=True,
            )

    def upload_snapshot_to_hf(self, cutoff_date: date) -> str | None:
        """Upload a local snapshot to HuggingFace Hub.

        Uses the same repo as the main model but with branch name 'snapshot-{date}'.

        Args:
            cutoff_date: The snapshot cutoff date to upload

        Returns:
            HF repo ID if successful, None if HF not configured or snapshot doesn't exist
        """
        repo_id = self._get_hf_repo()
        if not repo_id:
            logger.warning(
                f"HF repo not configured for {self.forecaster_type}, skipping upload"
            )
            return None

        if not self.snapshot_exists(cutoff_date):
            logger.warning(
                f"Snapshot {cutoff_date} does not exist locally, cannot upload"
            )
            return None

        token = self._get_hf_token()
        api = HfApi(token=token)
        self._ensure_hf_repo_exists(api, repo_id)

        snapshot_dir = self._snapshot_path(cutoff_date)
        branch_name = self._snapshot_branch_name(cutoff_date)

        logger.info(
            f"Uploading {self.forecaster_type} snapshot {cutoff_date} "
            f"to {repo_id} (branch: {branch_name})"
        )

        # Create the snapshot branch first (huggingface_hub 0.21+ requires explicit branch creation)
        try:
            api.create_branch(
                repo_id=repo_id,
                repo_type="model",
                branch=branch_name,
            )
            logger.info(f"Created branch {branch_name} on {repo_id}")
        except Exception as e:
            # Branch may already exist, which is fine
            if (
                "already exists" not in str(e).lower()
                and "reference already exists" not in str(e).lower()
            ):
                logger.warning(f"Could not create branch {branch_name}: {e}")

        api.upload_folder(
            folder_path=str(snapshot_dir),
            repo_id=repo_id,
            repo_type="model",
            revision=branch_name,
            commit_message=f"Add {self.forecaster_type} snapshot for {cutoff_date}",
        )

        return repo_id

    def download_snapshot_from_hf(self, cutoff_date: date) -> bool:
        """Download a snapshot from HuggingFace Hub if not available locally.

        Args:
            cutoff_date: The snapshot cutoff date to download

        Returns:
            True if snapshot is now available locally (downloaded or already existed),
            False if download failed or HF not configured
        """
        # Already have it locally
        if self.snapshot_exists(cutoff_date):
            return True

        repo_id = self._get_hf_repo()
        if not repo_id:
            logger.debug(f"HF repo not configured for {self.forecaster_type}")
            return False

        token = self._get_hf_token()
        branch_name = self._snapshot_branch_name(cutoff_date)

        try:
            logger.info(
                f"Downloading {self.forecaster_type} snapshot {cutoff_date} "
                f"from {repo_id} (branch: {branch_name})"
            )

            # Download to a temp location first
            local_dir = snapshot_download(
                repo_id=repo_id,
                revision=branch_name,
                repo_type="model",
                token=token,
            )

            # Copy files to our local snapshot path
            snapshot_dir = self._snapshot_path(cutoff_date)
            snapshot_dir.mkdir(parents=True, exist_ok=True)

            src_path = Path(local_dir)
            for file_name in [
                "weights.pt",
                "feature_scaler.pkl",
                "config.json",
                "metadata.json",
            ]:
                src_file = src_path / file_name
                if src_file.exists():
                    dst_file = snapshot_dir / file_name
                    # Copy file content
                    dst_file.write_bytes(src_file.read_bytes())

            logger.info(
                f"Successfully downloaded snapshot {cutoff_date} to {snapshot_dir}"
            )
            return True

        except Exception as e:
            logger.warning(f"Failed to download snapshot {cutoff_date} from HF: {e}")
            return False

    def ensure_snapshot_available(self, cutoff_date: date) -> bool:
        """Ensure a snapshot is available locally, downloading from HF if needed.

        This is the main method to use when loading snapshots - it will:
        1. Return True immediately if snapshot exists locally
        2. Try to download from HF if not available locally
        3. Return False if neither local nor HF has the snapshot

        Args:
            cutoff_date: The snapshot cutoff date

        Returns:
            True if snapshot is available locally (after potential download),
            False otherwise
        """
        if self.snapshot_exists(cutoff_date):
            return True

        # Try downloading from HF
        return self.download_snapshot_from_hf(cutoff_date)

    def list_hf_snapshots(self) -> list[date]:
        """List all snapshot branches available on HuggingFace Hub.

        Returns:
            List of cutoff dates for available snapshots on HF
        """
        repo_id = self._get_hf_repo()
        if not repo_id:
            return []

        token = self._get_hf_token()
        api = HfApi(token=token)

        try:
            refs = api.list_repo_refs(repo_id=repo_id, repo_type="model")
            snapshots = []

            for branch in refs.branches:
                if branch.name.startswith("snapshot-"):
                    try:
                        date_str = branch.name.replace("snapshot-", "")
                        snapshots.append(date.fromisoformat(date_str))
                    except ValueError:
                        continue

            return sorted(snapshots)
        except Exception as e:
            logger.warning(f"Failed to list HF snapshots: {e}")
            return []

    def sync_all_local_to_hf(self) -> list[date]:
        """Upload all local snapshots that aren't on HF yet.

        Returns:
            List of cutoff dates that were uploaded
        """
        local_snapshots = set(self.list_snapshots())
        hf_snapshots = set(self.list_hf_snapshots())

        to_upload = local_snapshots - hf_snapshots
        uploaded = []

        for cutoff_date in sorted(to_upload):
            if self.upload_snapshot_to_hf(cutoff_date):
                uploaded.append(cutoff_date)

        return uploaded

    def sync_all_hf_to_local(self) -> list[date]:
        """Download all HF snapshots that aren't local yet.

        Returns:
            List of cutoff dates that were downloaded
        """
        local_snapshots = set(self.list_snapshots())
        hf_snapshots = set(self.list_hf_snapshots())

        to_download = hf_snapshots - local_snapshots
        downloaded = []

        for cutoff_date in sorted(to_download):
            if self.download_snapshot_from_hf(cutoff_date):
                downloaded.append(cutoff_date)

        return downloaded


def create_snapshot_metadata(
    forecaster_type: str,
    cutoff_date: date,
    data_window_start: str,
    data_window_end: str,
    symbols: list[str],
    config: Any,
    train_loss: float,
    val_loss: float,
) -> dict[str, Any]:
    """Create metadata dictionary for a forecaster snapshot.

    Args:
        forecaster_type: "lstm" or "patchtst"
        cutoff_date: Data cutoff date
        data_window_start: Training data start date (ISO format)
        data_window_end: Training data end date (ISO format)
        symbols: List of symbols used
        config: Model configuration
        train_loss: Training loss
        val_loss: Validation loss

    Returns:
        Metadata dictionary.
    """
    return {
        "forecaster_type": forecaster_type,
        "cutoff_date": cutoff_date.isoformat(),
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
        },
    }

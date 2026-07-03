"""Local filesystem storage for forecaster snapshots.

Stores yearly LSTM and PatchTST model snapshots for walk-forward
forecast generation during RL training.

Also supports HuggingFace Hub upload/download using the same repo
as the main model but with branch naming convention:
- Main model: v2025-01-05-abc123
- Snapshots (hashed folder + branch): snapshot-2024-12-31-{12-hex-config-symbols-hash}
"""

import json
import logging
import pickle
import shutil
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch

if TYPE_CHECKING:
    from brain_api.storage.policy import StoragePolicy
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError
from sklearn.preprocessing import StandardScaler

from brain_api.core.config import (
    get_hf_lstm_halal_new_model_repo,
    get_hf_patchtst_halal_new_model_repo,
    get_hf_patchtst_nifty_shariah_500_model_repo,
    get_hf_token,
)
from brain_api.storage.base import DEFAULT_DATA_PATH
from brain_api.storage.forecaster_snapshots.artifacts import (
    LSTMSnapshotArtifacts,
    PatchTSTSnapshotArtifacts,
)
from brain_api.storage.forecaster_snapshots.snapshot_layout import (
    parse_hashed_snapshot_folder_name,
    snapshot_branch_basename,
)

logger = logging.getLogger(__name__)


class SnapshotLocalStorage:
    """Local hashed snapshot storage (sibling to main model versions).

    Layout: ``{base_path}/models/{forecaster_type}/snapshot-{cutoff}-{digest}/``
    with ``weights.pt``, ``feature_scaler.pkl``, ``config.json``, ``metadata.json``.

    The digest is twelve hex characters from :func:`~brain_api.core.version.compute_model_hash`.
    Legacy ``snapshot-{date}/`` dirs (no digest) are ignored.

    HuggingFace branches reuse the same basename as the local folder.
    """

    # Backwards-compat aliases mapping the legacy short forecaster name
    # (still passed by SAC walk-forward code) to the canonical
    # ``{model}_{universe}`` bucket name. SAC's only US bucket today
    # (``sac_halal_filtered``) trains on the top-15 of ``halal_new``, so
    # its forecaster snapshots come from the ``halal_new`` LSTM/PatchTST
    # buckets. When SAC gets a second universe (e.g. ``halal_india``)
    # this mapping will need to become bucket-aware -- for now keep it
    # explicit so a wrong universe surfaces as a key error rather than
    # silently writing to the wrong directory (AGENTS.md rule #1).
    _LEGACY_FORECASTER_ALIASES: ClassVar[dict[str, str]] = {
        "lstm": "lstm_halal_new",
        "patchtst": "patchtst_halal_new",
    }

    def __init__(
        self,
        forecaster_type: str,
        base_path: Path | str | None = None,
        hf_token: str | None = None,
    ):
        """Initialize storage.

        Args:
            forecaster_type: Bucket name in ``{model}_{universe}`` form
                (e.g. ``"lstm_halal_new"``, ``"patchtst_halal_new"``,
                ``"patchtst_nifty_shariah_500"``). This is also the
                disk subdirectory under ``data/models/`` so snapshots
                live as siblings of main model versions for that
                bucket. Distinct buckets MUST never share a folder.
                Legacy short names (``"lstm"``, ``"patchtst"``) are
                accepted as aliases for the corresponding
                ``halal_new`` bucket -- this matches the SAC
                walk-forward entrypoint that still passes the short
                name and points to the only forecaster bucket SAC
                currently consumes.
            base_path: Base path for data storage. Defaults to 'data/'.
            hf_token: HuggingFace API token. If None, uses HF_TOKEN env var.
        """
        if base_path is None:
            base_path = DEFAULT_DATA_PATH
        self.base_path = Path(base_path)
        self.forecaster_type = self._LEGACY_FORECASTER_ALIASES.get(
            forecaster_type, forecaster_type
        )
        if self.forecaster_type not in {
            "lstm_halal_new",
            "patchtst_halal_new",
            "patchtst_nifty_shariah_500",
        }:
            raise ValueError(
                f"Unknown forecaster snapshot bucket {forecaster_type!r}. "
                "Expected one of: lstm_halal_new, patchtst_halal_new, "
                "patchtst_nifty_shariah_500 (or legacy aliases 'lstm', "
                "'patchtst')."
            )
        self._hf_token = hf_token
        # Models directory where both main versions and snapshots live as siblings
        self._models_path = self.base_path / "models" / self.forecaster_type
        self._hf_missing: set[str] = set()

    def _get_hf_repo(self) -> str | None:
        """Get the HuggingFace repo ID for this forecaster bucket."""
        if self.forecaster_type == "lstm_halal_new":
            return get_hf_lstm_halal_new_model_repo()
        if self.forecaster_type == "patchtst_nifty_shariah_500":
            return get_hf_patchtst_nifty_shariah_500_model_repo()
        if self.forecaster_type == "patchtst_halal_new":
            return get_hf_patchtst_halal_new_model_repo()
        raise ValueError(
            f"Unknown snapshot forecaster_type: {self.forecaster_type!r}. "
            "Expected one of: lstm_halal_new, patchtst_halal_new, "
            "patchtst_nifty_shariah_500."
        )

    def _get_hf_token(self) -> str | None:
        """Get HF token from instance or environment."""
        return self._hf_token or get_hf_token()

    def _snapshot_branch_name(self, cutoff_date: date, snapshot_digest: str) -> str:
        """HF branch name and local subdirectory basename for a hashed snapshot."""
        return snapshot_branch_basename(cutoff_date, snapshot_digest)

    def _snapshot_path(self, cutoff_date: date, snapshot_digest: str) -> Path:
        """Path to hashed snapshot folder under this bucket's ``models`` directory."""
        return self._models_path / self._snapshot_branch_name(
            cutoff_date, snapshot_digest
        )

    def hashed_snapshot_dirs_for_cutoff(self, cutoff_date: date) -> list[Path]:
        """Return sorted hashed snapshot dirs for ``snapshot-{cutoff}-*`` pattern."""
        if not self._models_path.exists():
            return []
        return sorted(
            p
            for p in self._models_path.glob(f"snapshot-{cutoff_date.isoformat()}-*")
            if p.is_dir() and parse_hashed_snapshot_folder_name(p.name) is not None
        )

    def resolve_snapshot_directory(self, cutoff_date: date) -> Path:
        """Resolve exactly one hashed snapshot directory for ``cutoff_date``.

        Raises:
            ValueError: No matching hashed snapshot folder.
            RuntimeError: Multiple matching folders for the same cutoff.
        """
        matches = self.hashed_snapshot_dirs_for_cutoff(cutoff_date)
        if len(matches) == 0:
            raise ValueError(
                f"No hashed snapshot folder for cutoff {cutoff_date} under "
                f"{self._models_path}"
            )
        if len(matches) > 1:
            raise RuntimeError(
                f"Multiple snapshot folders for cutoff {cutoff_date}: {matches}"
            )
        return matches[0]

    def snapshot_exists(self, cutoff_date: date, snapshot_digest: str) -> bool:
        """Check if a hashed snapshot exists locally."""
        return self._snapshot_path(cutoff_date, snapshot_digest).exists()

    def snapshot_exists_anywhere(
        self,
        cutoff_date: date,
        snapshot_digest: str,
        *,
        check_hf: bool = False,
    ) -> bool:
        """Whether ``snapshot-{cutoff}-{digest}`` exists locally or on HF."""

        if self.snapshot_exists(cutoff_date, snapshot_digest):
            return True

        if check_hf:
            return self.snapshot_digest_exists_on_hf(cutoff_date, snapshot_digest)

        return False

    def snapshot_digest_exists_on_hf(
        self, cutoff_date: date, snapshot_digest: str
    ) -> bool:
        branch = self._snapshot_branch_name(cutoff_date, snapshot_digest)
        return branch in self._list_hf_hashed_snapshot_branch_names()

    def list_local_snapshot_identities(self) -> list[tuple[date, str]]:
        """Pairs ``(cutoff_date, snapshot_digest)`` for every local hashed snapshot."""
        if not self._models_path.exists():
            return []

        identities: list[tuple[date, str]] = []
        for entry in self._models_path.iterdir():
            if entry.is_dir():
                parsed = parse_hashed_snapshot_folder_name(entry.name)
                if parsed is not None:
                    identities.append(parsed)
        return sorted(identities, key=lambda t: (t[0], t[1]))

    def list_snapshots(self) -> list[date]:
        """Sorted unique cutoff dates with at least one hashed local snapshot."""

        cutoffs = {c for c, _ in self.list_local_snapshot_identities()}
        return sorted(cutoffs)

    def _list_hf_hashed_snapshot_branch_names(self) -> set[str]:
        repo_id = self._get_hf_repo()
        if not repo_id:
            return set()

        token = self._get_hf_token()
        api = HfApi(token=token)

        try:
            refs = api.list_repo_refs(repo_id=repo_id, repo_type="model")
        except Exception as e:
            logger.warning(f"Failed to list HF snapshot branches: {e}")
            return set()

        return {
            branch.name
            for branch in refs.branches
            if parse_hashed_snapshot_folder_name(branch.name) is not None
        }

    def list_hf_snapshot_identities(self) -> list[tuple[date, str]]:
        """HF snapshot identities from hashed branches only (legacy branches ignored)."""
        out: list[tuple[date, str]] = []
        for branch in sorted(self._list_hf_hashed_snapshot_branch_names()):
            parsed = parse_hashed_snapshot_folder_name(branch)
            if parsed is not None:
                out.append(parsed)
        return sorted(out)

    def write_snapshot(
        self,
        cutoff_date: date,
        snapshot_digest: str,
        model: Any,
        feature_scaler: StandardScaler,
        config: Any,
        metadata: dict[str, Any],
    ) -> Path:
        """Write a snapshot for ``snapshot-{cutoff}-{digest}/``.

        Removes sibling hashed folders for the same ``cutoff_date`` whose digest differs
        (keeps exactly one hashed layout per cutoff). Legacy ``snapshot-{cutoff}/`` dirs
        are untouched.
        """

        snapshot_dir = self._snapshot_path(cutoff_date, snapshot_digest)
        for stale in self._models_path.glob(f"snapshot-{cutoff_date.isoformat()}-*"):
            try:
                if stale.resolve() != snapshot_dir.resolve():
                    shutil.rmtree(stale)
            except OSError as exc:
                logger.warning(f"Could not remove stale snapshot dir {stale}: {exc}")

        snapshot_dir.mkdir(parents=True, exist_ok=True)

        weights_path = snapshot_dir / "weights.pt"
        torch.save(model.state_dict(), weights_path)

        scaler_path = snapshot_dir / "feature_scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(feature_scaler, f)

        config_path = snapshot_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config.to_dict(), f, indent=2)

        metadata_path = snapshot_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        branch_key = self._snapshot_branch_name(cutoff_date, snapshot_digest)
        self._hf_missing.discard(branch_key)

        return snapshot_dir

    def load_snapshot(
        self, cutoff_date: date
    ) -> LSTMSnapshotArtifacts | PatchTSTSnapshotArtifacts:
        """Load the hashed snapshot folder for ``cutoff_date``.

        Exactly one matching ``snapshot-{cutoff}-{digest}/`` must exist.
        """

        snapshot_dir = self.resolve_snapshot_directory(cutoff_date)

        config_path = snapshot_dir / "config.json"
        with open(config_path) as f:
            config_dict = json.load(f)

        scaler_path = snapshot_dir / "feature_scaler.pkl"
        with open(scaler_path, "rb") as f:
            feature_scaler = pickle.load(f)

        weights_path = snapshot_dir / "weights.pt"

        if self.forecaster_type.startswith("lstm"):
            from brain_api.core.lstm import LSTMConfig, LSTMModel

            config = LSTMConfig(**config_dict)
            model = LSTMModel(config)
            model.load_state_dict(
                torch.load(weights_path, weights_only=True, map_location="cpu")
            )
            model.eval()

            return LSTMSnapshotArtifacts(
                config=config,
                feature_scaler=feature_scaler,
                model=model,
                cutoff_date=cutoff_date,
            )
        from transformers import PatchTSTForPrediction

        from brain_api.core.patchtst import PatchTSTConfig

        config = PatchTSTConfig(**config_dict)
        hf_config = config.to_hf_config()
        model = PatchTSTForPrediction(hf_config)
        model.load_state_dict(
            torch.load(weights_path, weights_only=True, map_location="cpu")
        )
        model.eval()

        return PatchTSTSnapshotArtifacts(
            config=config,
            feature_scaler=feature_scaler,
            model=model,
            cutoff_date=cutoff_date,
        )

    def read_metadata(self, cutoff_date: date) -> dict[str, Any] | None:
        """Read metadata for a snapshot (hashed folder only).

        Legacy ``snapshot-{date}/`` is not consulted.
        """
        matches = self.hashed_snapshot_dirs_for_cutoff(cutoff_date)
        if len(matches) != 1:
            return None

        metadata_path = matches[0] / "metadata.json"
        if not metadata_path.exists():
            return None
        with open(metadata_path) as f:
            return json.load(f)

    def get_snapshot_for_year(self, year: int) -> date | None:
        """Get the snapshot cutoff date to use for predictions in a given year.

        For year N, we need a snapshot trained on data up to Dec 31 of year N-1.
        """

        target_cutoff = date(year - 1, 12, 31)

        if self.hashed_snapshot_dirs_for_cutoff(target_cutoff):
            return target_cutoff

        available = self.list_snapshots()
        valid_snapshots = [d for d in available if d <= target_cutoff]

        if valid_snapshots:
            return max(valid_snapshots)

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

    def upload_snapshot_to_hf(
        self, cutoff_date: date, snapshot_digest: str
    ) -> str | None:
        """Upload a hashed local snapshot branch ``snapshot-{date}-{digest}``."""

        repo_id = self._get_hf_repo()
        if not repo_id:
            logger.warning(
                f"HF repo not configured for {self.forecaster_type}, skipping upload"
            )
            return None

        if not self.snapshot_exists(cutoff_date, snapshot_digest):
            logger.warning(
                f"Snapshot {cutoff_date}/{snapshot_digest} does not exist locally, "
                "cannot upload"
            )
            return None

        token = self._get_hf_token()
        api = HfApi(token=token)
        self._ensure_hf_repo_exists(api, repo_id)

        snapshot_dir = self._snapshot_path(cutoff_date, snapshot_digest)
        branch_name = self._snapshot_branch_name(cutoff_date, snapshot_digest)

        logger.info(
            f"Uploading {self.forecaster_type} snapshot {branch_name} "
            f"to {repo_id} (branch: {branch_name})"
        )

        try:
            api.create_branch(
                repo_id=repo_id,
                repo_type="model",
                branch=branch_name,
            )
            logger.info(f"Created branch {branch_name} on {repo_id}")
        except Exception as e:
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
            commit_message=(
                f"Add {self.forecaster_type} snapshot {cutoff_date} ({snapshot_digest})"
            ),
        )

        # Clean up any stale branches for this cutoff date on HF
        try:
            stale_prefix = f"snapshot-{cutoff_date.isoformat()}-"
            for branch in self._list_hf_hashed_snapshot_branch_names():
                if branch.startswith(stale_prefix) and branch != branch_name:
                    logger.info(f"Deleting stale HF branch: {branch}")
                    try:
                        api.delete_branch(
                            repo_id=repo_id, repo_type="model", branch=branch
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to delete stale HF branch {branch}: {e}"
                        )
        except Exception as e:
            logger.warning(
                f"Failed to clean up stale HF branches for {cutoff_date}: {e}"
            )

        return repo_id

    def download_snapshot_from_hf(
        self, cutoff_date: date, snapshot_digest: str
    ) -> bool:
        """Download ``snapshot-{cutoff}-{digest}`` from HuggingFace if missing locally."""

        if self.snapshot_exists(cutoff_date, snapshot_digest):
            return True

        repo_id = self._get_hf_repo()
        if not repo_id:
            logger.debug(f"HF repo not configured for {self.forecaster_type}")
            return False

        token = self._get_hf_token()
        branch_name = self._snapshot_branch_name(cutoff_date, snapshot_digest)

        if branch_name in self._hf_missing:
            return False

        try:
            logger.info(
                f"Downloading {self.forecaster_type} snapshot {branch_name} "
                f"from {repo_id}"
            )

            local_dir = snapshot_download(
                repo_id=repo_id,
                revision=branch_name,
                repo_type="model",
                token=token,
            )

            snapshot_dir = self._snapshot_path(cutoff_date, snapshot_digest)

            # Remove conflicting sibling hashed dirs for same cutoff before copy.
            for stale in self._models_path.glob(
                f"snapshot-{cutoff_date.isoformat()}-*"
            ):
                try:
                    if stale.resolve() != snapshot_dir.resolve():
                        shutil.rmtree(stale)
                except OSError as exc:
                    logger.warning(
                        f"Could not remove stale snapshot dir {stale}: {exc}"
                    )

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
                    dst_file.write_bytes(src_file.read_bytes())

            logger.info(
                f"Successfully downloaded snapshot {branch_name} to {snapshot_dir}"
            )
            self._hf_missing.discard(branch_name)
            return True

        except Exception as e:
            logger.warning(f"Failed to download snapshot {branch_name} from HF: {e}")
            self._hf_missing.add(branch_name)
            return False

    def ensure_snapshot_available(
        self,
        cutoff_date: date,
        snapshot_digest: str,
        policy: "StoragePolicy | None" = None,
    ) -> bool:
        """Ensure ``snapshot-{cutoff}-{digest}`` is available locally (HF fallback)."""

        from brain_api.storage.policy import (
            StoragePolicy,
            StoragePolicyError,
            get_storage_policy,
        )

        if policy is None:
            policy = get_storage_policy()

        branch_name = self._snapshot_branch_name(cutoff_date, snapshot_digest)

        if self.snapshot_exists(cutoff_date, snapshot_digest):
            return True

        hf_repo = self._get_hf_repo()
        if policy is StoragePolicy.HF_FIRST and not hf_repo:
            raise StoragePolicyError(
                f"hf_first policy requires HF repo for snapshot bucket "
                f"{self.forecaster_type!r}; got none."
            )

        if not hf_repo:
            return False

        if branch_name in self._hf_missing:
            return False

        return self.download_snapshot_from_hf(cutoff_date, snapshot_digest)

    def list_hf_snapshots(self) -> list[date]:
        """Sorted unique cutoff dates present on HF as hashed snapshot branches."""

        return sorted({c for c, _ in self.list_hf_snapshot_identities()})

    def sync_all_local_to_hf(self) -> list[tuple[date, str]]:
        """Upload local hashed snapshots whose branch is missing on HF."""

        local_ident = set(self.list_local_snapshot_identities())
        hf_ident = set(self.list_hf_snapshot_identities())

        uploaded: list[tuple[date, str]] = []
        for cutoff, digest in sorted(local_ident - hf_ident):
            if self.upload_snapshot_to_hf(cutoff, digest):
                uploaded.append((cutoff, digest))

        return uploaded

    def sync_all_hf_to_local(self) -> list[tuple[date, str]]:
        """Download HF hashed snapshots missing locally."""

        local_ident = set(self.list_local_snapshot_identities())
        hf_ident = set(self.list_hf_snapshot_identities())

        downloaded: list[tuple[date, str]] = []
        for cutoff, digest in sorted(hf_ident - local_ident):
            if self.download_snapshot_from_hf(cutoff, digest):
                downloaded.append((cutoff, digest))

        return downloaded

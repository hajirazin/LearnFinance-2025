"""HuggingFace upload/download for hashed forecaster snapshots."""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError

from brain_api.core.training_utils import evaluate_forecaster_artifact_health
from brain_api.storage.forecaster_snapshots.snapshot_files import (
    copy_snapshot_artifacts,
    evict_sibling_hashed_snapshot_dirs,
    snapshot_train_val_losses,
)

if TYPE_CHECKING:
    from brain_api.storage.policy import StoragePolicy

logger = logging.getLogger(__name__)


class SnapshotHFMixin:
    """HF branch upload/download. Mixed into :class:`SnapshotLocalStorage`."""

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

            src_path = Path(local_dir)
            metadata_path = src_path / "metadata.json"
            if not metadata_path.exists():
                logger.warning(
                    f"Downloaded snapshot {branch_name} has no metadata.json; "
                    "refusing to install"
                )
                self._hf_missing.add(branch_name)
                return False
            with open(metadata_path) as f:
                downloaded_metadata = json.load(f)
            losses = snapshot_train_val_losses(downloaded_metadata)
            if losses is None:
                logger.warning(
                    f"Downloaded snapshot {branch_name} metadata is missing "
                    "finite train/val loss; refusing to install"
                )
                self._hf_missing.add(branch_name)
                return False
            health = evaluate_forecaster_artifact_health(
                train_loss=losses[0],
                val_loss=losses[1],
                baseline_loss=None,
                artifact_dir=None,
            )
            if not health.is_healthy:
                logger.warning(
                    f"Downloaded snapshot {branch_name} failed health check "
                    f"({health.failure_reasons}); refusing to install"
                )
                self._hf_missing.add(branch_name)
                return False

            snapshot_dir = self._snapshot_path(cutoff_date, snapshot_digest)
            evict_sibling_hashed_snapshot_dirs(
                self._models_path, cutoff_date, snapshot_dir
            )
            copy_snapshot_artifacts(src_path, snapshot_dir)

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
        policy: StoragePolicy | None = None,
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

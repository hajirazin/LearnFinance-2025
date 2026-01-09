"""Helper functions for training routes.

This module provides reusable helper functions that are shared across
multiple training routes (LSTM, PatchTST, PPO, SAC).
"""

import logging
from dataclasses import dataclass
from typing import Any

from brain_api.core.config import get_storage_backend
from brain_api.storage.base_local import BaseLocalModelStorage

logger = logging.getLogger(__name__)


@dataclass
class PriorVersionInfo:
    """Information about the prior model version.

    Used to determine whether a newly trained model should be promoted.
    """

    version: str | None
    metadata: dict[str, Any] | None
    val_loss: float | None


def get_prior_version_info(
    local_storage: BaseLocalModelStorage,
    hf_storage_class: type | None = None,
    hf_model_repo: str | None = None,
) -> PriorVersionInfo:
    """Get prior version info from local storage, falling back to HF if needed.

    This function handles the GCP + HF scenario where local storage is ephemeral.
    When local storage has no current version, it checks HuggingFace for the
    current model metadata (if STORAGE_BACKEND=hf and HF repo is configured).

    Args:
        local_storage: Local storage instance (e.g., LocalModelStorage, PatchTSTModelStorage)
        hf_storage_class: HuggingFace storage class (e.g., HuggingFaceModelStorage).
                         Required for HF fallback.
        hf_model_repo: HuggingFace repo ID (e.g., 'username/learnfinance-lstm').
                      Required for HF fallback.

    Returns:
        PriorVersionInfo with version, metadata, and val_loss (if available)
    """
    # Try local first
    prior_version = local_storage.read_current_version()
    prior_metadata = None
    prior_val_loss = None

    if prior_version:
        logger.info(f"Found local prior version: {prior_version}")
        prior_metadata = local_storage.read_metadata(prior_version)
        if prior_metadata:
            prior_val_loss = prior_metadata.get("metrics", {}).get("val_loss")
            logger.info(f"Prior val_loss from local: {prior_val_loss}")
    elif get_storage_backend() == "hf" and hf_storage_class and hf_model_repo:
        # Check HF for prior version if local is empty
        logger.info(f"No local version found, checking HuggingFace: {hf_model_repo}")
        try:
            hf_storage = hf_storage_class(repo_id=hf_model_repo)
            prior_metadata = hf_storage.get_current_metadata()
            if prior_metadata:
                prior_version = prior_metadata.get("version")
                prior_val_loss = prior_metadata.get("metrics", {}).get("val_loss")
                logger.info(
                    f"Found HF prior version: {prior_version}, val_loss: {prior_val_loss}"
                )
        except Exception as e:
            logger.warning(f"Failed to get prior version from HF: {e}")
            # HF not available, proceed without prior

    return PriorVersionInfo(
        version=prior_version,
        metadata=prior_metadata,
        val_loss=prior_val_loss,
    )

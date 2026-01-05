"""HuggingFace Hub storage for PatchTST model artifacts.

Note: PatchTST HuggingFace storage is planned for future implementation.
For now, PatchTST models are stored locally only.
"""

import logging

logger = logging.getLogger(__name__)


class HuggingFaceModelStorage:
    """HuggingFace Hub storage for PatchTST model artifacts.

    Note: This is a stub implementation. PatchTST HuggingFace upload
    will be implemented when needed.
    """

    def __init__(
        self,
        repo_id: str | None = None,
        token: str | None = None,
    ):
        """Initialize HuggingFace model storage.

        Args:
            repo_id: HuggingFace repo ID for PatchTST models.
            token: HuggingFace API token.
        """
        self.repo_id = repo_id
        self.token = token
        logger.warning(
            "PatchTST HuggingFace storage is not yet implemented. "
            "Models are stored locally only."
        )

    def upload_model(self, *args, **kwargs):
        """Upload PatchTST model to HuggingFace Hub.

        Note: Not yet implemented.
        """
        raise NotImplementedError(
            "PatchTST HuggingFace upload is not yet implemented. "
            "Use local storage for now."
        )

    def download_model(self, *args, **kwargs):
        """Download PatchTST model from HuggingFace Hub.

        Note: Not yet implemented.
        """
        raise NotImplementedError(
            "PatchTST HuggingFace download is not yet implemented. "
            "Use local storage for now."
        )


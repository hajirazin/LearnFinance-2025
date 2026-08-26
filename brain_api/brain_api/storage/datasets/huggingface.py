"""HuggingFace Hub storage for remaining sentiment datasets (twitter only)."""

import logging

import pandas as pd
from huggingface_hub import HfApi

from brain_api.core.config import get_hf_token, get_hf_twitter_sentiment_repo

logger = logging.getLogger(__name__)


class HuggingFaceDatasetStorage:
    """HuggingFace Hub storage for twitter sentiment datasets."""

    def __init__(
        self,
        twitter_repo_id: str | None = None,
        token: str | None = None,
    ):
        self.twitter_repo_id = twitter_repo_id or get_hf_twitter_sentiment_repo()
        self.token = token or get_hf_token()
        self.api = HfApi(token=self.token)

    def push_twitter_sentiment(
        self,
        df: pd.DataFrame,
        commit_message: str = "Update twitter sentiment data",
    ) -> None:
        del df, commit_message
        logger.warning("Twitter sentiment storage is not implemented.")

    def load_twitter_sentiment(self) -> None:
        logger.warning("Twitter sentiment storage is not implemented.")

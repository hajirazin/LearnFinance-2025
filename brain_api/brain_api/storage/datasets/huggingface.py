"""HuggingFace Hub storage for sentiment datasets."""

import logging
import tempfile
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError

from brain_api.core.config import (
    get_hf_news_sentiment_repo,
    get_hf_token,
    get_hf_twitter_sentiment_repo,
)

logger = logging.getLogger(__name__)


class HuggingFaceDatasetStorage:
    """HuggingFace Hub storage for sentiment datasets.

    Stores sentiment data as parquet files in HuggingFace Dataset repositories.
    Supports news sentiment now, with twitter sentiment planned for future.

    Authentication:
        The huggingface_hub library automatically uses credentials from:
        1. Explicit token parameter (if provided)
        2. HF_TOKEN environment variable
        3. Cached token from `huggingface-cli login` (~/.cache/huggingface/token)

        Recommended: Run `huggingface-cli login` once, then no token needed.
    """

    def __init__(
        self,
        news_repo_id: str | None = None,
        twitter_repo_id: str | None = None,
        token: str | None = None,
    ):
        """Initialize HuggingFace dataset storage.

        Args:
            news_repo_id: HF repo for news sentiment. Defaults to HF_NEWS_SENTIMENT_REPO.
            twitter_repo_id: HF repo for twitter sentiment. Defaults to HF_TWITTER_SENTIMENT_REPO.
            token: HuggingFace API token. If None, uses HF_TOKEN env var or
                   cached token from `huggingface-cli login`.
        """
        self.news_repo_id = news_repo_id or get_hf_news_sentiment_repo()
        self.twitter_repo_id = twitter_repo_id or get_hf_twitter_sentiment_repo()
        self.token = token or get_hf_token()  # None is OK - HfApi uses cached token
        self.api = HfApi(token=self.token)

    def _ensure_dataset_repo_exists(self, repo_id: str) -> None:
        """Create the HF dataset repo if it doesn't exist."""
        try:
            self.api.repo_info(repo_id=repo_id, repo_type="dataset")
        except RepositoryNotFoundError:
            logger.info(f"Creating HuggingFace dataset repo: {repo_id}")
            self.api.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                exist_ok=True,
            )

    def push_news_sentiment(
        self,
        df: pd.DataFrame,
        commit_message: str = "Update news sentiment data",
    ) -> str:
        """Push news sentiment DataFrame to HuggingFace Datasets.

        Args:
            df: DataFrame with sentiment data (date, symbol, sentiment_score, etc.)
            commit_message: Git commit message for the upload

        Returns:
            URL of the uploaded dataset
        """
        if not self.news_repo_id:
            raise ValueError(
                "News sentiment repo not configured. "
                "Set HF_NEWS_SENTIMENT_REPO environment variable."
            )

        self._ensure_dataset_repo_exists(self.news_repo_id)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Save as parquet
            parquet_path = tmppath / "data" / "daily_sentiment.parquet"
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(parquet_path, index=False)

            # Create dataset card README
            readme_path = tmppath / "README.md"
            readme_content = f"""---
tags:
- finance
- sentiment
- news
- learnfinance
---

# LearnFinance News Sentiment Dataset

Daily aggregated news sentiment scores for stocks in the halal universe.

## Dataset Details

- **Date Range**: {df['date'].min()} to {df['date'].max()}
- **Symbols**: {df['symbol'].nunique()} unique stocks
- **Records**: {len(df):,} rows

## Schema

| Column | Type | Description |
|--------|------|-------------|
| date | date | Trading date |
| symbol | string | Stock ticker |
| sentiment_score | float | Aggregated sentiment [-1, 1] |
| article_count | int | Number of articles |
| avg_confidence | float | Average FinBERT confidence |

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{self.news_repo_id}")
df = dataset["train"].to_pandas()
```
"""
            with open(readme_path, "w") as f:
                f.write(readme_content)

            logger.info(f"Uploading news sentiment to {self.news_repo_id}")

            self.api.upload_folder(
                folder_path=tmpdir,
                repo_id=self.news_repo_id,
                repo_type="dataset",
                commit_message=commit_message,
            )

        return f"https://huggingface.co/datasets/{self.news_repo_id}"

    def load_news_sentiment(self) -> pd.DataFrame | None:
        """Load news sentiment data from HuggingFace Datasets.

        Returns:
            DataFrame with sentiment data, or None if not available.
        """
        if not self.news_repo_id:
            logger.warning("News sentiment repo not configured")
            return None

        try:
            from datasets import load_dataset

            logger.info(f"Loading news sentiment from {self.news_repo_id}")
            dataset = load_dataset(self.news_repo_id, token=self.token)

            # Convert to pandas
            if "train" in dataset:
                return dataset["train"].to_pandas()
            else:
                # Try first available split
                first_split = next(iter(dataset.keys()))
                return dataset[first_split].to_pandas()

        except Exception as e:
            logger.error(f"Failed to load news sentiment: {e}")
            return None

    def push_twitter_sentiment(
        self,
        df: pd.DataFrame,
        commit_message: str = "Update twitter sentiment data",
    ) -> None:
        """Push twitter sentiment DataFrame to HuggingFace Datasets.

        Note: Twitter sentiment is planned for future implementation.

        Args:
            df: DataFrame with twitter sentiment data
            commit_message: Git commit message for the upload

        Returns:
            None - feature not yet implemented
        """
        logger.warning(
            "Twitter sentiment storage is planned for future implementation. "
            "This method currently does nothing."
        )
        return None

    def load_twitter_sentiment(self) -> None:
        """Load twitter sentiment data from HuggingFace Datasets.

        Note: Twitter sentiment is planned for future implementation.

        Returns:
            None - feature not yet implemented
        """
        logger.warning(
            "Twitter sentiment storage is planned for future implementation. "
            "This method currently returns None."
        )
        return None


"""Publication policy for the news-sentiment parquet."""

import logging
from pathlib import Path

from brain_api.etl.config import get_hf_news_sentiment_repo

logger = logging.getLogger(__name__)


def publish_sentiment_parquet(*, parquet_path: Path, local_only: bool) -> str | None:
    """Publish the current parquet unless explicitly requested to stay local."""
    if local_only:
        return None
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Cannot publish sentiment data; parquet not found: {parquet_path}"
        )
    return upload_to_huggingface(parquet_path)


def upload_to_huggingface(parquet_path: Path) -> str:
    """Upload the sentiment parquet and return its Hugging Face dataset URL."""
    hf_repo = get_hf_news_sentiment_repo()
    if not hf_repo:
        raise RuntimeError("HF_NEWS_SENTIMENT_REPO is required for publication")

    try:
        from huggingface_hub import HfApi

        logger.info("Uploading to HuggingFace: %s", hf_repo)
        api = HfApi()
        try:
            api.repo_info(repo_id=hf_repo, repo_type="dataset")
        except Exception:
            logger.info("Creating HuggingFace repository: %s", hf_repo)
            api.create_repo(repo_id=hf_repo, repo_type="dataset", exist_ok=True)

        api.upload_file(
            path_or_fileobj=str(parquet_path),
            path_in_repo="data/daily_sentiment.parquet",
            repo_id=hf_repo,
            repo_type="dataset",
        )
    except Exception as exc:
        logger.error("HuggingFace upload failed: %s", exc)
        raise RuntimeError(f"HuggingFace upload failed: {exc}") from exc

    hf_url = f"https://huggingface.co/datasets/{hf_repo}"
    logger.info("Uploaded to %s", hf_url)
    return hf_url

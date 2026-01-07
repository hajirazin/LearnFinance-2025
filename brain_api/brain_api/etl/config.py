"""Configuration for news sentiment ETL pipeline.

Moved from news_sentiment_etl/core/config.py.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

# Environment variable for HF news sentiment dataset repo
ENV_HF_NEWS_SENTIMENT_REPO = "HF_NEWS_SENTIMENT_REPO"


def get_hf_news_sentiment_repo() -> str | None:
    """Get HuggingFace news sentiment dataset repository from environment."""
    return os.environ.get(ENV_HF_NEWS_SENTIMENT_REPO)


@dataclass
class ETLConfig:
    """Configuration for the news sentiment ETL pipeline.

    Attributes:
        dataset_name: HuggingFace dataset identifier
        data_files: Glob pattern for dataset files (None = all)
        data_input_dir: Directory for downloaded dataset files
        output_dir: Directory for output files
        checkpoint_dir: Directory for processing checkpoints
        cache_dir: Directory for sentiment cache database
        output_parquet: Filename for Parquet output (None to skip)
        output_csv: Filename for CSV output (None to skip)
        batch_size: Number of articles to process per FinBERT batch
        checkpoint_interval: Save checkpoint every N batches
        sentiment_threshold: Min |p_pos - p_neg| to include article (bounded filter)
        use_gpu: Whether to use GPU for FinBERT (auto-detect if None)
        max_articles: Maximum NEW articles to score (None = all, useful for testing)
        filter_to_halal: Whether to filter output to halal universe only
        hf_token: HuggingFace token for gated datasets (optional)
        local_only: If True, skip HF upload even if HF_NEWS_SENTIMENT_REPO is set
    """

    # Dataset configuration
    dataset_name: str = "Brianferrell787/financial-news-multisource"
    data_files: str | None = "data/*/*.parquet"
    data_input_dir: Path = field(default_factory=lambda: Path("data/input"))

    # Output configuration
    output_dir: Path = field(default_factory=lambda: Path("data/output"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("data/checkpoints"))
    cache_dir: Path = field(default_factory=lambda: Path("data/cache"))
    output_parquet: str | None = "daily_sentiment.parquet"  # None to skip
    output_csv: str | None = None  # None to skip (default: parquet only)

    # Processing configuration
    batch_size: int = 256  # FinBERT batch size
    checkpoint_interval: int = 100  # Save every N batches
    sentiment_threshold: float = 0.1  # Bounded filter threshold
    use_gpu: bool | None = None  # Auto-detect if None
    max_articles: int | None = None  # Limit for testing

    # Universe filtering
    filter_to_halal: bool = True

    # Authentication
    hf_token: str | None = None

    # HuggingFace upload control
    local_only: bool = False  # If True, skip HF upload

    def __post_init__(self) -> None:
        """Ensure paths are Path objects."""
        if isinstance(self.data_input_dir, str):
            self.data_input_dir = Path(self.data_input_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.checkpoint_dir, str):
            self.checkpoint_dir = Path(self.checkpoint_dir)
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)


# FinBERT model configuration (shared with brain_api.core.finbert)
FINBERT_MODEL = "ProsusAI/finbert"
FINBERT_MAX_LENGTH = 512

# Dataset subsets that have stock symbols in extra_fields
SUBSETS_WITH_SYMBOLS = [
    "benzinga_6000stocks",
    "bloomberg_reuters",
    "fnspid_news",
    "finsen_us_2007_2023",
    "yahoo_finance_felixdrinkall",
    "reddit_finance_sp500",
    "yahoo_finance_articles",
]

# All available subsets in the dataset
ALL_SUBSETS = [
    "benzinga_6000stocks",
    "bloomberg_reuters",
    "sentarl_combined",
    "fnspid_news",
    "mind_news_2019",
    "gold_news_kaggle",
    "huffpost_news",
    "sp500_daily_headlines",
    "reddit_worldnews_2008_2016",
    "djia_stock_headlines",
    "wikinews_articles",
    "headlines_10sites_2007_2022",
    "nyt_headlines_1990_2020",
    "nyt_headlines_2010_2021",
    "cnbc_headlines",
    "american_news_jonasbecker",
    "us_financial_news_2018_jeet",
    "nyt_articles_2000_present",
    "finsen_us_2007_2023",
    "yahoo_finance_felixdrinkall",
    "reddit_finance_sp500",
    "frontpage_news",
    "yahoo_finance_articles",
    "all_the_news_2",
]




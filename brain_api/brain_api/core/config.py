"""Configuration for LSTM training and storage backends."""

import os
from datetime import date

from dateutil.relativedelta import relativedelta

# Environment variable names
ENV_LSTM_LOOKBACK_YEARS = "LSTM_TRAIN_LOOKBACK_YEARS"
ENV_LSTM_WINDOW_END_DATE = "LSTM_TRAIN_WINDOW_END_DATE"

# HuggingFace Hub environment variables
ENV_HF_TOKEN = "HF_TOKEN"
ENV_HF_LSTM_MODEL_REPO = "HF_LSTM_MODEL_REPO"  # LSTM forecaster
ENV_HF_PATCHTST_MODEL_REPO = "HF_PATCHTST_MODEL_REPO"  # PatchTST forecaster
ENV_HF_PPO_LSTM_MODEL_REPO = "HF_PPO_LSTM_MODEL_REPO"  # PPO + LSTM allocator
ENV_HF_PPO_PATCHTST_MODEL_REPO = "HF_PPO_PATCHTST_MODEL_REPO"  # PPO + PatchTST allocator
ENV_HF_NEWS_SENTIMENT_REPO = "HF_NEWS_SENTIMENT_REPO"
ENV_HF_TWITTER_SENTIMENT_REPO = "HF_TWITTER_SENTIMENT_REPO"
ENV_STORAGE_BACKEND = "STORAGE_BACKEND"

# Alpaca News API environment variables
ENV_ALPACA_API_KEY = "ALPACA_API_KEY"
ENV_ALPACA_API_SECRET = "ALPACA_API_SECRET"

# Defaults
DEFAULT_LOOKBACK_YEARS = 15
DEFAULT_STORAGE_BACKEND = "local"  # Options: "local", "hf"


def get_hf_token() -> str | None:
    """Get HuggingFace API token from environment."""
    return os.environ.get(ENV_HF_TOKEN)


def get_hf_lstm_model_repo() -> str | None:
    """Get HuggingFace LSTM model repository name (e.g., 'username/learnfinance-lstm')."""
    return os.environ.get(ENV_HF_LSTM_MODEL_REPO)


def get_hf_patchtst_model_repo() -> str | None:
    """Get HuggingFace PatchTST model repository name."""
    return os.environ.get(ENV_HF_PATCHTST_MODEL_REPO)


def get_hf_ppo_lstm_model_repo() -> str | None:
    """Get HuggingFace PPO + LSTM model repository name."""
    return os.environ.get(ENV_HF_PPO_LSTM_MODEL_REPO)


def get_hf_ppo_patchtst_model_repo() -> str | None:
    """Get HuggingFace PPO + PatchTST model repository name."""
    return os.environ.get(ENV_HF_PPO_PATCHTST_MODEL_REPO)


def get_hf_news_sentiment_repo() -> str | None:
    """Get HuggingFace news sentiment dataset repository name."""
    return os.environ.get(ENV_HF_NEWS_SENTIMENT_REPO)


def get_hf_twitter_sentiment_repo() -> str | None:
    """Get HuggingFace twitter sentiment dataset repository name."""
    return os.environ.get(ENV_HF_TWITTER_SENTIMENT_REPO)


def get_storage_backend() -> str:
    """Get the storage backend to use ('local' or 'hf')."""
    return os.environ.get(ENV_STORAGE_BACKEND, DEFAULT_STORAGE_BACKEND)


def resolve_training_window() -> tuple[date, date]:
    """Resolve the training data window from API config/environment.

    Reads:
    - LSTM_TRAIN_LOOKBACK_YEARS: number of years to look back (default: 10)
    - LSTM_TRAIN_WINDOW_END_DATE: optional override for end date (YYYY-MM-DD)

    Returns:
        Tuple of (start_date, end_date) as date objects.
    """
    # Get lookback years from env or use default
    lookback_str = os.environ.get(ENV_LSTM_LOOKBACK_YEARS, "")
    lookback_years = int(lookback_str) if lookback_str else DEFAULT_LOOKBACK_YEARS

    # Get end date from env or use today
    end_date_str = os.environ.get(ENV_LSTM_WINDOW_END_DATE, "")
    end_date = date.fromisoformat(end_date_str) if end_date_str else date.today()

    # Compute start date
    start_date = end_date - relativedelta(years=lookback_years)

    return start_date, end_date


def get_alpaca_api_key() -> str:
    """Get Alpaca API key from environment."""
    return os.environ.get(ENV_ALPACA_API_KEY, "")


def get_alpaca_api_secret() -> str:
    """Get Alpaca API secret from environment."""
    return os.environ.get(ENV_ALPACA_API_SECRET, "")

"""Configuration for LSTM training and storage backends."""

import os
from datetime import date, timedelta
from enum import Enum

# Environment variable names
ENV_LSTM_LOOKBACK_YEARS = "LSTM_TRAIN_LOOKBACK_YEARS"
ENV_LSTM_WINDOW_END_DATE = "LSTM_TRAIN_WINDOW_END_DATE"
ENV_LSTM_TRAIN_UNIVERSE = "LSTM_TRAIN_UNIVERSE"
ENV_CUTOFF_DATE = "CUTOFF_DATE"

# HuggingFace Hub environment variables
ENV_HF_TOKEN = "HF_TOKEN"
ENV_HF_LSTM_MODEL_REPO = "HF_LSTM_MODEL_REPO"  # LSTM forecaster
ENV_HF_PATCHTST_MODEL_REPO = "HF_PATCHTST_MODEL_REPO"  # PatchTST forecaster
ENV_HF_PPO_MODEL_REPO = "HF_PPO_MODEL_REPO"  # PPO allocator (dual forecasts)
ENV_HF_SAC_MODEL_REPO = "HF_SAC_MODEL_REPO"  # SAC allocator (dual forecasts)
ENV_HF_NEWS_SENTIMENT_REPO = "HF_NEWS_SENTIMENT_REPO"
ENV_HF_TWITTER_SENTIMENT_REPO = "HF_TWITTER_SENTIMENT_REPO"
ENV_STORAGE_BACKEND = "STORAGE_BACKEND"

# Alpaca News API environment variables
ENV_ALPACA_API_KEY = "ALPACA_API_KEY"
ENV_ALPACA_API_SECRET = "ALPACA_API_SECRET"

# Defaults
DEFAULT_LOOKBACK_YEARS = 15
DEFAULT_STORAGE_BACKEND = "local"  # Options: "local", "hf"


class UniverseType(str, Enum):
    """Stock universe types for training models.

    Each universe represents a different set of stocks to train on.
    Using str as base class allows direct string comparison and serialization.
    """

    HALAL = "halal"  # Halal ETF universe (~45 stocks from SPUS/HLAL/SPTE)
    SP500 = "sp500"  # S&P 500 (~500 stocks from datahub.io)
    # Future universes:
    # INDIAN_HALAL = "indian_halal"
    # NIFTY500 = "nifty500"


DEFAULT_LSTM_TRAIN_UNIVERSE = UniverseType.SP500


def get_hf_token() -> str | None:
    """Get HuggingFace API token from environment."""
    return os.environ.get(ENV_HF_TOKEN)


def get_hf_lstm_model_repo() -> str | None:
    """Get HuggingFace LSTM model repository name (e.g., 'username/learnfinance-lstm')."""
    return os.environ.get(ENV_HF_LSTM_MODEL_REPO)


def get_hf_patchtst_model_repo() -> str | None:
    """Get HuggingFace PatchTST model repository name."""
    return os.environ.get(ENV_HF_PATCHTST_MODEL_REPO)


def get_hf_ppo_model_repo() -> str | None:
    """Get HuggingFace PPO model repository name (unified with dual forecasts)."""
    return os.environ.get(ENV_HF_PPO_MODEL_REPO)


def get_hf_sac_model_repo() -> str | None:
    """Get HuggingFace SAC model repository name (unified with dual forecasts)."""
    return os.environ.get(ENV_HF_SAC_MODEL_REPO)


def get_hf_news_sentiment_repo() -> str | None:
    """Get HuggingFace news sentiment dataset repository name."""
    return os.environ.get(ENV_HF_NEWS_SENTIMENT_REPO)


def get_hf_twitter_sentiment_repo() -> str | None:
    """Get HuggingFace twitter sentiment dataset repository name."""
    return os.environ.get(ENV_HF_TWITTER_SENTIMENT_REPO)


def get_storage_backend() -> str:
    """Get the storage backend to use ('local' or 'hf')."""
    return os.environ.get(ENV_STORAGE_BACKEND, DEFAULT_STORAGE_BACKEND)


def get_lstm_train_universe() -> UniverseType:
    """Get LSTM training universe from environment.

    Controls which stock universe LSTM is trained on. Different universes
    produce different model versions (symbols are included in version hash).

    Returns:
        UniverseType enum value.

    Raises:
        ValueError: If LSTM_TRAIN_UNIVERSE env var has an invalid value.
    """
    env_value = os.environ.get(ENV_LSTM_TRAIN_UNIVERSE, "")
    if not env_value:
        return DEFAULT_LSTM_TRAIN_UNIVERSE

    try:
        return UniverseType(env_value.lower())
    except ValueError as err:
        valid_options = [e.value for e in UniverseType]
        raise ValueError(
            f"Invalid LSTM_TRAIN_UNIVERSE='{env_value}'. Valid options: {valid_options}"
        ) from err


def resolve_cutoff_date(reference_date: date | None = None) -> date:
    """Resolve cutoff date to the Friday BEFORE the reference date.

    The cutoff is ALWAYS the previous Friday, even if reference_date is Friday.
    This ensures we have complete week data before making predictions.

    Args:
        reference_date: Base date. If None, reads from CUTOFF_DATE env var or uses today.

    Returns:
        The most recent Friday strictly before reference_date.

    Examples:
        - Monday Jan 12 -> Friday Jan 9
        - Friday Jan 9 -> Friday Jan 2 (previous Friday, not same day)
        - Saturday Jan 10 -> Friday Jan 9
        - Sunday Jan 11 -> Friday Jan 9
    """
    if reference_date is None:
        env_date = os.environ.get(ENV_CUTOFF_DATE, "")
        reference_date = date.fromisoformat(env_date) if env_date else date.today()

    # weekday(): Mon=0, Tue=1, Wed=2, Thu=3, Fri=4, Sat=5, Sun=6
    days_since_friday = (reference_date.weekday() - 4) % 7
    if days_since_friday == 0:
        # reference_date is Friday, go back to previous Friday
        return reference_date - timedelta(days=7)
    return reference_date - timedelta(days=days_since_friday)


def resolve_training_window() -> tuple[date, date]:
    """Resolve the training data window from API config/environment.

    Reads:
    - LSTM_TRAIN_LOOKBACK_YEARS: number of years to look back (default: 15)
    - LSTM_TRAIN_WINDOW_END_DATE: optional override for end date (YYYY-MM-DD)

    Returns:
        Tuple of (start_date, end_date) where end_date is always a Friday.
        Start date is anchored to January 1st of (end_year - lookback_years).
    """
    # Get lookback years from env or use default
    lookback_str = os.environ.get(ENV_LSTM_LOOKBACK_YEARS, "")
    lookback_years = int(lookback_str) if lookback_str else DEFAULT_LOOKBACK_YEARS

    # Get reference date from env or use today, then anchor to Friday
    end_date_str = os.environ.get(ENV_LSTM_WINDOW_END_DATE, "")
    reference_date = date.fromisoformat(end_date_str) if end_date_str else None
    end_date = resolve_cutoff_date(reference_date)

    # Compute start date (anchored to January 1st of year)
    start_date = date(end_date.year - lookback_years, 1, 1)

    return start_date, end_date


def get_alpaca_api_key() -> str:
    """Get Alpaca API key from environment."""
    return os.environ.get(ENV_ALPACA_API_KEY, "")


def get_alpaca_api_secret() -> str:
    """Get Alpaca API secret from environment."""
    return os.environ.get(ENV_ALPACA_API_SECRET, "")

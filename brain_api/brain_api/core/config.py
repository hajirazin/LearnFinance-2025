"""Configuration for forecaster training and storage backends."""

import os
from datetime import date, timedelta
from enum import Enum

# Environment variable names
ENV_LSTM_LOOKBACK_YEARS = "LSTM_TRAIN_LOOKBACK_YEARS"
ENV_LSTM_WINDOW_END_DATE = "LSTM_TRAIN_WINDOW_END_DATE"
ENV_ETL_UNIVERSE = "ETL_UNIVERSE"
ENV_CUTOFF_DATE = "CUTOFF_DATE"

# HuggingFace Hub environment variables
#
# Per-bucket HF repos. The naming convention is
# ``HF_{MODEL}_{UNIVERSE}_MODEL_REPO`` so that adding a new bucket
# (e.g. ``sac_halal`` or ``sac_halal_india`` for an A/B comparison)
# is one new env var without disturbing existing buckets. Each bucket
# has an independent ``current`` pointer on HF; promoting one MUST NOT
# touch another.
ENV_HF_TOKEN = "HF_TOKEN"
ENV_HF_LSTM_HALAL_NEW_MODEL_REPO = "HF_LSTM_HALAL_NEW_MODEL_REPO"
ENV_HF_PATCHTST_HALAL_NEW_MODEL_REPO = "HF_PATCHTST_HALAL_NEW_MODEL_REPO"
ENV_HF_PATCHTST_NIFTY_SHARIAH_500_MODEL_REPO = (
    "HF_PATCHTST_NIFTY_SHARIAH_500_MODEL_REPO"
)
ENV_HF_SAC_HALAL_FILTERED_MODEL_REPO = "HF_SAC_HALAL_FILTERED_MODEL_REPO"
ENV_HF_NEWS_SENTIMENT_REPO = "HF_NEWS_SENTIMENT_REPO"
ENV_HF_TWITTER_SENTIMENT_REPO = "HF_TWITTER_SENTIMENT_REPO"
ENV_STORAGE_BACKEND = "STORAGE_BACKEND"

# Alpaca News API environment variables
ENV_ALPACA_API_KEY = "ALPACA_API_KEY"
ENV_ALPACA_API_SECRET = "ALPACA_API_SECRET"

# Defaults
DEFAULT_LOOKBACK_YEARS = 10
DEFAULT_STORAGE_BACKEND = "local"  # Options: "local", "hf"


class UniverseType(str, Enum):
    """Stock universe types for training models and ETL.

    Each universe represents a different set of stocks to train on.
    Using str as base class allows direct string comparison and serialization.

    Universe selection for *training* now flows through the per-bucket
    registry (see ``brain_api.core.model_buckets``) instead of an env
    var, so a workflow can A/B-test two universes against the same
    endpoint concurrently. ``UniverseType`` is retained for ETL
    selection only; the training endpoints accept the universe as a
    request body field instead.
    """

    HALAL = "halal"  # Halal ETF universe (~14 stocks from SPUS/HLAL/SPTE)
    SP500 = "sp500"  # S&P 500 (~500 stocks from datahub.io)
    HALAL_NEW = "halal_new"  # Expanded halal (~410 stocks from 5 ETFs + Alpaca filter)
    HALAL_FILTERED = "halal_filtered"  # Top 15 factor-scored from halal_new
    HALAL_INDIA = "halal_india"  # Top 15 PatchTST-scored from Nifty 500 Shariah (NSE)
    NIFTY_SHARIAH_500 = "nifty_shariah_500"  # All ~210 Nifty 500 Shariah constituents


DEFAULT_ETL_UNIVERSE = UniverseType.HALAL_FILTERED


def get_hf_token() -> str | None:
    """Get HuggingFace API token from environment."""
    return os.environ.get(ENV_HF_TOKEN)


def get_hf_lstm_halal_new_model_repo() -> str | None:
    """Get HF repo for LSTM trained on the ``halal_new`` universe."""
    return os.environ.get(ENV_HF_LSTM_HALAL_NEW_MODEL_REPO)


def get_hf_patchtst_halal_new_model_repo() -> str | None:
    """Get HF repo for PatchTST trained on the ``halal_new`` universe."""
    return os.environ.get(ENV_HF_PATCHTST_HALAL_NEW_MODEL_REPO)


def get_hf_patchtst_nifty_shariah_500_model_repo() -> str | None:
    """Get HF repo for PatchTST trained on the ``nifty_shariah_500`` universe (India)."""
    return os.environ.get(ENV_HF_PATCHTST_NIFTY_SHARIAH_500_MODEL_REPO)


def get_hf_sac_halal_filtered_model_repo() -> str | None:
    """Get HF repo for SAC trained on the ``halal_filtered`` universe."""
    return os.environ.get(ENV_HF_SAC_HALAL_FILTERED_MODEL_REPO)


def get_hf_news_sentiment_repo() -> str | None:
    """Get HuggingFace news sentiment dataset repository name."""
    return os.environ.get(ENV_HF_NEWS_SENTIMENT_REPO)


def get_hf_twitter_sentiment_repo() -> str | None:
    """Get HuggingFace twitter sentiment dataset repository name."""
    return os.environ.get(ENV_HF_TWITTER_SENTIMENT_REPO)


def get_storage_backend() -> str:
    """Get the storage backend to use ('local' or 'hf')."""
    return os.environ.get(ENV_STORAGE_BACKEND, DEFAULT_STORAGE_BACKEND)


def get_etl_universe() -> UniverseType:
    """Get ETL pipeline universe from environment.

    Controls which stock universe the news-sentiment ETL and
    sentiment-gaps pipelines filter to.

    Returns:
        UniverseType enum value.

    Raises:
        ValueError: If ETL_UNIVERSE env var has an invalid value.
    """
    env_value = os.environ.get(ENV_ETL_UNIVERSE, "")
    if not env_value:
        return DEFAULT_ETL_UNIVERSE

    try:
        return UniverseType(env_value.lower())
    except ValueError as err:
        valid_options = [e.value for e in UniverseType]
        raise ValueError(
            f"Invalid ETL_UNIVERSE='{env_value}'. Valid options: {valid_options}"
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
    - LSTM_TRAIN_LOOKBACK_YEARS: number of years to look back (default: 10)
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

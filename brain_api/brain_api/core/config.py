"""Configuration for LSTM training."""

import os
from datetime import date
from dateutil.relativedelta import relativedelta


# Environment variable names
ENV_LSTM_LOOKBACK_YEARS = "LSTM_TRAIN_LOOKBACK_YEARS"
ENV_LSTM_WINDOW_END_DATE = "LSTM_TRAIN_WINDOW_END_DATE"

# Defaults
DEFAULT_LOOKBACK_YEARS = 10


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
    if end_date_str:
        end_date = date.fromisoformat(end_date_str)
    else:
        end_date = date.today()

    # Compute start date
    start_date = end_date - relativedelta(years=lookback_years)

    return start_date, end_date



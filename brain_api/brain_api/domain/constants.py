"""Domain constants for brain_api.

This module centralizes all magic numbers and configuration constants
to improve maintainability and make the codebase more self-documenting.
"""

# ============================================================================
# LSTM Training Constants
# ============================================================================

# Sequence length for LSTM input (trading days of history)
LSTM_SEQUENCE_LENGTH = 60

# Number of OHLCV features
LSTM_INPUT_SIZE = 5  # open, high, low, close, volume

# Model architecture defaults
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.2

# Training defaults
LSTM_BATCH_SIZE = 32
LSTM_LEARNING_RATE = 0.001
LSTM_EPOCHS = 50
LSTM_VALIDATION_SPLIT = 0.2

# Trading days in a year (used for lookback calculations)
TRADING_DAYS_PER_YEAR = 252

# Default lookback years for training data
DEFAULT_LOOKBACK_YEARS = 3

# Minimum trading days in a week to be valid
MIN_WEEK_TRADING_DAYS = 3


# ============================================================================
# Sentiment Constants
# ============================================================================

# Decay constant for recency weighting (in days)
# Determines how quickly older articles lose influence
SENTIMENT_TAU_DAYS = 7.0

# Minimum articles required for reliable sentiment
MIN_ARTICLES_FOR_SENTIMENT = 3

# Minimum sentiment score magnitude to be considered non-neutral
MIN_SENTIMENT_MAGNITUDE = 0.01

# FinBERT model
FINBERT_MODEL = "ProsusAI/finbert"
FINBERT_MAX_LENGTH = 512


# ============================================================================
# API Limits
# ============================================================================

# Maximum symbols per request (news sentiment)
MAX_NEWS_SYMBOLS = 50

# Maximum articles to fetch per symbol
MAX_ARTICLES_PER_SYMBOL = 30

# Default articles per symbol
DEFAULT_ARTICLES_PER_SYMBOL = 30

# Maximum top articles to return in response
MAX_TOP_K_ARTICLES = 10

# Default top articles to return
DEFAULT_TOP_K_ARTICLES = 10

# Maximum symbols for fundamentals endpoints
MAX_FUNDAMENTALS_SYMBOLS = 20

# Maximum symbols for historical sentiment
MAX_HISTORICAL_SENTIMENT_SYMBOLS = 20

# Alpha Vantage daily API limit (free tier)
ALPHA_VANTAGE_DAILY_LIMIT = 25


# ============================================================================
# HRP Allocation Constants
# ============================================================================

# Default lookback days for HRP
HRP_DEFAULT_LOOKBACK_DAYS = 252

# Minimum data days required for a symbol
HRP_MIN_DATA_DAYS = 60


# ============================================================================
# Inference Constants
# ============================================================================

# Direction classification thresholds (in percentage points)
DIRECTION_UP_THRESHOLD = 0.5  # > 0.5% = UP
DIRECTION_DOWN_THRESHOLD = -0.5  # < -0.5% = DOWN


# ============================================================================
# Storage Constants
# ============================================================================

# Default data directory
DEFAULT_DATA_PATH = "data"

# Model storage paths
MODELS_SUBDIR = "models"
LSTM_MODELS_SUBDIR = "lstm"
CURRENT_VERSION_FILENAME = "current"

# Raw data storage
RAW_DATA_SUBDIR = "raw"
FEATURES_SUBDIR = "features"

# Cache storage
CACHE_SUBDIR = "cache"


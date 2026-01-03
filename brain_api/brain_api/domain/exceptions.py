"""Custom exceptions for brain_api domain.

This module defines domain-specific exceptions to replace generic exceptions
and provide better error handling and debugging.
"""


class BrainAPIError(Exception):
    """Base exception for all brain_api errors."""

    pass


# ============================================================================
# Data errors
# ============================================================================


class DataError(BrainAPIError):
    """Base class for data-related errors."""

    pass


class InsufficientDataError(DataError):
    """Raised when there's not enough data to perform an operation.

    Examples:
    - Not enough price history for LSTM inference
    - Too few articles for sentiment aggregation
    - Insufficient trading days in a week
    """

    def __init__(self, message: str, required: int | None = None, available: int | None = None):
        super().__init__(message)
        self.required = required
        self.available = available


class DataNotFoundError(DataError):
    """Raised when required data cannot be found.

    Examples:
    - Model artifacts not found
    - Price data not available for symbol
    - Parquet file missing
    """

    def __init__(self, message: str, resource: str | None = None):
        super().__init__(message)
        self.resource = resource


class DataValidationError(DataError):
    """Raised when data fails validation.

    Examples:
    - Invalid date format
    - Empty symbol list
    - Out-of-range values
    """

    def __init__(self, message: str, field: str | None = None, value: any = None):
        super().__init__(message)
        self.field = field
        self.value = value


# ============================================================================
# Model errors
# ============================================================================


class ModelError(BrainAPIError):
    """Base class for model-related errors."""

    pass


class ModelNotFoundError(ModelError):
    """Raised when a required model is not found.

    Examples:
    - No current LSTM model version
    - Model artifacts missing
    """

    def __init__(self, message: str, model_type: str | None = None, version: str | None = None):
        super().__init__(message)
        self.model_type = model_type
        self.version = version


class ModelLoadError(ModelError):
    """Raised when a model fails to load.

    Examples:
    - Corrupted model weights
    - Incompatible model version
    """

    def __init__(self, message: str, model_path: str | None = None):
        super().__init__(message)
        self.model_path = model_path


class InferenceError(ModelError):
    """Raised when model inference fails.

    Examples:
    - Feature dimension mismatch
    - Invalid input data
    """

    pass


class TrainingError(ModelError):
    """Raised when model training fails.

    Examples:
    - NaN loss during training
    - Memory allocation failure
    """

    pass


# ============================================================================
# External service errors
# ============================================================================


class ExternalServiceError(BrainAPIError):
    """Base class for external service errors."""

    pass


class RateLimitError(ExternalServiceError):
    """Raised when an API rate limit is exceeded.

    Examples:
    - Alpha Vantage daily limit reached
    - yfinance throttling
    """

    def __init__(self, message: str, service: str, limit: int | None = None, reset_at: str | None = None):
        super().__init__(message)
        self.service = service
        self.limit = limit
        self.reset_at = reset_at


class FetchError(ExternalServiceError):
    """Raised when data fetching from external service fails.

    Examples:
    - Network error fetching from yfinance
    - Invalid API response from Alpha Vantage
    """

    def __init__(self, message: str, service: str, symbol: str | None = None):
        super().__init__(message)
        self.service = service
        self.symbol = symbol


# ============================================================================
# Storage errors
# ============================================================================


class StorageError(BrainAPIError):
    """Base class for storage-related errors."""

    pass


class StorageReadError(StorageError):
    """Raised when reading from storage fails."""

    def __init__(self, message: str, path: str | None = None):
        super().__init__(message)
        self.path = path


class StorageWriteError(StorageError):
    """Raised when writing to storage fails."""

    def __init__(self, message: str, path: str | None = None):
        super().__init__(message)
        self.path = path


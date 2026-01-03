"""LSTM-related domain entities.

These are pure dataclasses with no external ML framework dependencies.
"""

from dataclasses import dataclass
from datetime import date
from typing import Any


@dataclass
class LSTMConfig:
    """LSTM model hyperparameters and training config.

    The LSTM predicts weekly returns (Mon open → Fri close), not daily prices.
    This aligns with the RL agent's weekly decision horizon.
    """

    # Model architecture
    input_size: int = 5  # OHLCV features (log returns)
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2

    # Forecast settings
    # Single output: weekly return = (fri_close - mon_open) / mon_open
    forecast_horizon: int = 1

    # Training
    sequence_length: int = 60  # 60 trading days lookback
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 50
    validation_split: float = 0.2

    # Feature engineering
    use_returns: bool = True  # Use log returns for input features (more stationary)

    # Week filtering
    min_week_days: int = 3  # Skip weeks with fewer than 3 trading days

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "forecast_horizon": self.forecast_horizon,
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "validation_split": self.validation_split,
            "use_returns": self.use_returns,
            "min_week_days": self.min_week_days,
        }


# Default configuration
DEFAULT_CONFIG = LSTMConfig()


@dataclass
class DatasetResult:
    """Result of dataset building for weekly return prediction.

    Note: Uses numpy ndarray and sklearn scaler, passed as Any to avoid
    tight coupling to specific implementations. The actual types are:
    - X: np.ndarray with shape (n_samples, seq_len, n_features)
    - y: np.ndarray with shape (n_samples, 1)
    - feature_scaler: sklearn.preprocessing.StandardScaler
    """

    X: Any  # np.ndarray - Input sequences
    y: Any  # np.ndarray - Targets (weekly returns)
    feature_scaler: Any  # StandardScaler - Fitted scaler for features


@dataclass
class TrainingResult:
    """Result of LSTM training for weekly return prediction.

    Note: model type is Any to avoid tight coupling to PyTorch.
    """

    model: Any  # LSTMModel (PyTorch nn.Module)
    feature_scaler: Any  # StandardScaler
    config: LSTMConfig
    train_loss: float
    val_loss: float
    baseline_loss: float  # Baseline: predict 0 return (no change)


@dataclass
class WeekBoundaries:
    """Trading week boundaries for inference.

    Represents the target week for prediction, computed with holiday awareness.
    """

    target_week_start: date  # First trading day of the week (Mon or later if holiday)
    target_week_end: date  # Last trading day of the week (Fri or earlier if holiday)
    calendar_monday: date  # Calendar Monday of the ISO week
    calendar_friday: date  # Calendar Friday of the ISO week


@dataclass
class InferenceFeatures:
    """Features prepared for inference for a single symbol.

    Note: features type is Any to avoid tight coupling to numpy.
    """

    symbol: str
    features: Any | None  # np.ndarray - Shape: (seq_len, n_features) or None
    has_enough_history: bool
    history_days_used: int
    data_end_date: date | None  # Last date of data used


@dataclass
class SymbolPrediction:
    """Prediction result for a single symbol."""

    symbol: str
    predicted_weekly_return_pct: float | None  # Percentage (e.g., 2.5 for +2.5%)
    direction: str  # "UP", "DOWN", or "FLAT"
    has_enough_history: bool
    history_days_used: int
    data_end_date: str | None  # ISO format
    target_week_start: str  # ISO format
    target_week_end: str  # ISO format


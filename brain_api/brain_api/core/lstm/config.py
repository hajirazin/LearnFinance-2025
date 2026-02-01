"""LSTM model configuration."""

from dataclasses import dataclass
from typing import Any


@dataclass
class LSTMConfig:
    """LSTM model hyperparameters and training config.

    The LSTM predicts next-day returns (close-to-close), enabling iterative
    5-day prediction for weekly forecasts. This allows computing both weekly
    return (compounded from daily) and volatility (std of daily returns).
    """

    # Model architecture
    input_size: int = 5  # OHLCV features (log returns)
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2

    # Forecast settings
    # Single output: next-day return (close-to-close)
    forecast_horizon: int = 1

    # Training
    sequence_length: int = 60  # 60 trading days lookback
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
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


DEFAULT_CONFIG = LSTMConfig()

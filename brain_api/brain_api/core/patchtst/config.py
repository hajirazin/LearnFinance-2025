"""PatchTST model configuration."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import PatchTSTConfig as HFPatchTSTConfig


@dataclass
class PatchTSTConfig:
    """PatchTST model hyperparameters and training config.

    PatchTST (Patch Time Series Transformer) configuration for multi-channel
    weekly return prediction. This model supports multiple input channels:
    - Price features (OHLCV log returns)
    - News sentiment features
    - Fundamental ratios

    The model predicts weekly returns aligned with the RL agent's weekly
    decision horizon (Mon open → Fri close).
    """

    # Model architecture
    num_input_channels: int = 11  # OHLCV (5) + News (1) + Fundamentals (5)
    context_length: int = 60  # 60 trading days lookback (same as LSTM)
    prediction_length: int = 1  # Single output: weekly return
    patch_length: int = 16  # Standard patch size for PatchTST
    stride: int = 8  # 50% overlap between patches

    # Transformer architecture
    d_model: int = 64  # Hidden dimension
    num_attention_heads: int = 4
    num_hidden_layers: int = 2
    ffn_dim: int = 128  # Feed-forward network dimension
    dropout: float = 0.2

    # Training
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    validation_split: float = 0.2

    # Feature engineering
    use_returns: bool = True  # Use log returns for OHLCV (more stationary)

    # Week filtering
    min_week_days: int = 3  # Skip weeks with fewer than 3 trading days

    # Feature channel names for documentation
    feature_names: list[str] = field(default_factory=lambda: [
        "open_ret", "high_ret", "low_ret", "close_ret", "volume_ret",  # OHLCV
        "news_sentiment",  # News
        "gross_margin", "operating_margin", "net_margin",  # Fundamentals
        "current_ratio", "debt_to_equity",
    ])

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "num_input_channels": self.num_input_channels,
            "context_length": self.context_length,
            "prediction_length": self.prediction_length,
            "patch_length": self.patch_length,
            "stride": self.stride,
            "d_model": self.d_model,
            "num_attention_heads": self.num_attention_heads,
            "num_hidden_layers": self.num_hidden_layers,
            "ffn_dim": self.ffn_dim,
            "dropout": self.dropout,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "validation_split": self.validation_split,
            "use_returns": self.use_returns,
            "min_week_days": self.min_week_days,
            "feature_names": self.feature_names,
        }

    def to_hf_config(self) -> "HFPatchTSTConfig":
        """Convert to HuggingFace PatchTSTConfig."""
        from transformers import PatchTSTConfig as HFPatchTSTConfig

        return HFPatchTSTConfig(
            num_input_channels=self.num_input_channels,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            patch_length=self.patch_length,
            patch_stride=self.stride,
            d_model=self.d_model,
            num_attention_heads=self.num_attention_heads,
            num_hidden_layers=self.num_hidden_layers,
            ffn_dim=self.ffn_dim,
            dropout=self.dropout,
            # Use distribution output for regression
            loss="mse",
            # Scaling for better training
            scaling="std",
        )


DEFAULT_CONFIG = PatchTSTConfig()


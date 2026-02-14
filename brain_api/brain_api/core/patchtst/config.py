"""PatchTST model configuration."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import PatchTSTConfig as HFPatchTSTConfig


@dataclass
class PatchTSTConfig:
    """PatchTST model hyperparameters and training config.

    5-channel OHLCV direct 5-day multi-task prediction with RevIN.

    Uses channel-independent architecture with shared Transformer weights.
    Multi-task loss on ALL 5 OHLCV channels provides data augmentation for
    the shared weights. At inference, close_ret predictions are extracted
    for the RL agent. RevIN (scaling="std") handles per-channel per-sample
    normalization internally.

    Input channels: OHLCV log returns (open_ret, high_ret, low_ret,
    close_ret, volume_ret).
    """

    # Model architecture
    num_input_channels: int = 5  # OHLCV log returns only
    context_length: int = 60  # 60 trading days lookback (same as LSTM)
    prediction_length: int = 5  # Direct 5-day prediction
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
    learning_rate: float = 0.0003
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 15  # Stop if val_loss doesn't improve for N epochs
    weight_decay: float = 1e-4  # L2 regularization
    max_grad_norm: float = 1.0  # Gradient clipping

    # Feature engineering
    use_returns: bool = True  # Use log returns for OHLCV (more stationary)

    # Week filtering
    min_week_days: int = 3  # Skip weeks with fewer than 3 trading days

    # Feature channel names (OHLCV only)
    feature_names: list[str] = field(
        default_factory=lambda: [
            "open_ret",
            "high_ret",
            "low_ret",
            "close_ret",
            "volume_ret",
        ]
    )

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
            "early_stopping_patience": self.early_stopping_patience,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "use_returns": self.use_returns,
            "min_week_days": self.min_week_days,
            "feature_names": self.feature_names,
        }

    def to_hf_config(self) -> "HFPatchTSTConfig":
        """Convert to HuggingFace PatchTSTConfig.

        IMPORTANT: This must match _create_patchtst_model() in training.py exactly
        to ensure model architecture consistency between training and inference.

        RevIN (scaling="std") is kept as default -- handles per-channel per-sample
        normalization internally. DO NOT set scaling=None.
        """
        from transformers import PatchTSTConfig as HFPatchTSTConfig

        return HFPatchTSTConfig(
            num_input_channels=self.num_input_channels,
            context_length=self.context_length,
            patch_length=self.patch_length,
            stride=self.stride,
            d_model=self.d_model,
            num_attention_heads=self.num_attention_heads,
            num_hidden_layers=self.num_hidden_layers,
            ffn_dim=self.ffn_dim,
            dropout=self.dropout,
            prediction_length=self.prediction_length,
            # RevIN defaults to scaling="std" -- handles per-channel normalization
            # Additional settings to match training
            attention_dropout=self.dropout,
            positional_dropout=self.dropout,
            use_cls_token=False,  # Use pooling instead
            pooling_type="mean",
        )


DEFAULT_CONFIG = PatchTSTConfig()

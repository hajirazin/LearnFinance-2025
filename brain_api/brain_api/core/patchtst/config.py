"""PatchTST model configuration."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import PatchTSTConfig as HFPatchTSTConfig


@dataclass
class PatchTSTConfig:
    """PatchTST model hyperparameters and training config.

    Locked research contract: one close-return channel, 10/5 geometry
    (11 unpadded patches), mean pooling, channel attention off.

    Training optimizes denormalized close_ret MSE so the objective matches
    Alpha-HRP / score-batch ranking on compounded close returns. RevIN
    (scaling="std") handles per-sample input normalization internally.

    Input channel: adjusted-close log return (``close_ret``).
    """

    # Model architecture
    num_input_channels: int = 1  # close_ret only
    context_length: int = 60  # 60 trading days lookback (same as LSTM)
    prediction_length: int = 5  # Direct 5-day prediction
    patch_length: int = 10  # Two trading weeks per patch
    stride: int = 5  # Advance one trading week; 50% overlap

    # Transformer architecture
    d_model: int = 64  # Hidden dimension
    num_attention_heads: int = 4
    num_hidden_layers: int = 2
    ffn_dim: int = 128  # Feed-forward network dimension
    dropout: float = 0.2

    # Training
    batch_size: int = 256
    learning_rate: float = 0.0003
    epochs: int = 60
    validation_split: float = 0.2
    early_stopping_patience: int = 8  # Stop if rank-IC checkpoint does not improve
    # E8: after Phase A close-only MSE, loss/grad ~100x smaller; wd=1e-4
    # dominated Adam updates (wd||theta|| / ||g|| ~ 40). Default 0 lets the model train.
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0  # Gradient clipping

    # Feature engineering
    use_returns: bool = True  # Use log returns (more stationary)

    # Week filtering
    min_week_days: int = 3  # Skip weeks with fewer than 3 trading days

    # Feature channel names (close-only)
    feature_names: list[str] = field(default_factory=lambda: ["close_ret"])

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

        Every material Hugging Face field is pinned to the locked research
        contract so defaults cannot drift. Hugging Face PatchTST has no generic
        ``dropout`` setting; the domain value is applied only to attention and
        positional dropout.
        """
        from transformers import PatchTSTConfig as HFPatchTSTConfig

        return HFPatchTSTConfig(
            num_input_channels=self.num_input_channels,
            context_length=self.context_length,
            distribution_output="student_t",
            loss="mse",
            patch_length=self.patch_length,
            patch_stride=self.stride,
            num_hidden_layers=self.num_hidden_layers,
            d_model=self.d_model,
            num_attention_heads=self.num_attention_heads,
            share_embedding=True,
            channel_attention=False,
            ffn_dim=self.ffn_dim,
            norm_type="batchnorm",
            norm_eps=1e-5,
            attention_dropout=self.dropout,
            positional_dropout=self.dropout,
            path_dropout=0.0,
            ff_dropout=0.0,
            bias=True,
            activation_function="gelu",
            pre_norm=True,
            positional_encoding_type="sincos",
            use_cls_token=False,
            init_std=0.02,
            share_projection=True,
            scaling="std",
            do_mask_input=False,
            pooling_type="mean",
            head_dropout=0.0,
            prediction_length=self.prediction_length,
        )


DEFAULT_CONFIG = PatchTSTConfig()

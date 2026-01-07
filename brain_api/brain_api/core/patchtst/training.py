"""PatchTST model training."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from transformers import PatchTSTConfig as HFPatchTSTConfig
from transformers import PatchTSTForPrediction

from brain_api.core.patchtst.config import PatchTSTConfig
from brain_api.core.training_utils import get_device


@dataclass
class TrainingResult:
    """Result of PatchTST training."""

    model: PatchTSTForPrediction
    feature_scaler: StandardScaler
    config: PatchTSTConfig
    train_loss: float
    val_loss: float
    baseline_loss: float


def _create_patchtst_model(config: PatchTSTConfig) -> PatchTSTForPrediction:
    """Create a HuggingFace PatchTST model from our config.

    Args:
        config: Our PatchTSTConfig

    Returns:
        Initialized PatchTSTForPrediction model
    """
    hf_config = HFPatchTSTConfig(
        num_input_channels=config.num_input_channels,
        context_length=config.context_length,
        patch_length=config.patch_length,
        stride=config.stride,
        d_model=config.d_model,
        num_attention_heads=config.num_attention_heads,
        num_hidden_layers=config.num_hidden_layers,
        ffn_dim=config.ffn_dim,
        dropout=config.dropout,
        prediction_length=config.prediction_length,
        # Additional settings
        attention_dropout=config.dropout,
        positional_dropout=config.dropout,
        use_cls_token=False,  # Use pooling instead
        pooling_type="mean",
    )
    return PatchTSTForPrediction(hf_config)


def train_model_pytorch(
    X: np.ndarray,
    y: np.ndarray,
    feature_scaler: StandardScaler,
    config: PatchTSTConfig,
) -> TrainingResult:
    """Train PatchTST model using PyTorch.

    Args:
        X: Input sequences, shape (n_samples, context_length, n_channels)
        y: Targets, shape (n_samples, 1) - weekly returns
        feature_scaler: Fitted scaler for input features
        config: Model configuration

    Returns:
        TrainingResult with trained model and metrics
    """
    device = get_device()
    print(f"[PatchTST] Training on device: {device}")

    if len(X) == 0:
        print("[PatchTST] No training data - returning dummy model")
        model = _create_patchtst_model(config)
        return TrainingResult(
            model=model,
            feature_scaler=feature_scaler,
            config=config,
            train_loss=float("inf"),
            val_loss=float("inf"),
            baseline_loss=float("inf"),
        )

    # Time-based train/val split
    split_idx = int(len(X) * (1 - config.validation_split))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"[PatchTST] Dataset: {len(X)} samples ({len(X_train)} train, {len(X_val)} val)")
    print(f"[PatchTST] Config: {config.epochs} epochs, batch_size={config.batch_size}, lr={config.learning_rate}")
    print(f"[PatchTST] Channels: {config.num_input_channels}, context_length={config.context_length}")

    # Convert to tensors
    # HuggingFace PatchTST expects shape (batch, sequence_length, num_channels)
    # Our X is already in that shape, no transpose needed
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)

    # Create model
    model = _create_patchtst_model(config).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training loop
    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = 0
    log_interval = max(1, config.epochs // 5)

    print("[PatchTST] Starting training...")

    for epoch in range(config.epochs):
        model.train()

        indices = torch.randperm(len(X_train_t), device=device)
        total_train_loss = 0.0
        n_batches = 0

        for i in range(0, len(indices), config.batch_size):
            batch_idx = indices[i : i + config.batch_size]
            batch_X = X_train_t[batch_idx]
            batch_y = y_train_t[batch_idx]

            optimizer.zero_grad()

            # PatchTST outputs prediction_outputs of shape (batch, pred_len, channels)
            # We want the mean prediction across channels for each sample
            outputs = model(past_values=batch_X).prediction_outputs
            # Take mean across channels and prediction length
            outputs = outputs.mean(dim=(1, 2)).unsqueeze(1)

            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_train_loss / n_batches

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(past_values=X_val_t).prediction_outputs
            val_outputs = val_outputs.mean(dim=(1, 2)).unsqueeze(1)
            val_loss = criterion(val_outputs, y_val_t).item()

        if (epoch + 1) % log_interval == 0 or epoch == 0:
            print(f"[PatchTST] Epoch {epoch + 1}/{config.epochs}: train_loss={avg_train_loss:.6f}, val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }

    print(f"[PatchTST] Best model at epoch {best_epoch} with val_loss={best_val_loss:.6f}")

    # Restore best model on CPU
    model_cpu = _create_patchtst_model(config)
    if best_model_state is not None:
        model_cpu.load_state_dict(best_model_state)

    # Final metrics
    model.eval()
    with torch.no_grad():
        train_outputs = model(past_values=X_train_t).prediction_outputs
        train_outputs = train_outputs.mean(dim=(1, 2)).unsqueeze(1)
        final_train_loss = criterion(train_outputs, y_train_t).item()

    # Baseline: predict 0 return
    baseline_loss = float(np.mean(y_val**2))

    print(f"[PatchTST] Training complete: train_loss={final_train_loss:.6f}, val_loss={best_val_loss:.6f}, baseline={baseline_loss:.6f}")
    beats_baseline = best_val_loss < baseline_loss
    print(f"[PatchTST] Model {'BEATS' if beats_baseline else 'does NOT beat'} baseline")

    return TrainingResult(
        model=model_cpu,
        feature_scaler=feature_scaler,
        config=config,
        train_loss=final_train_loss,
        val_loss=best_val_loss,
        baseline_loss=baseline_loss,
    )


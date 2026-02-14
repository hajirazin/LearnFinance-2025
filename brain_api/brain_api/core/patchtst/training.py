"""PatchTST model training.

5-channel OHLCV multi-task training with direct 5-day prediction.

Uses HuggingFace built-in loss which computes equal-weight MSE on ALL 5
channels in RevIN-normalized space. This is PatchTST's intended usage --
shared Transformer weights learn temporal patterns from 5 related OHLCV
signals (data augmentation effect).

X: (n_samples, context_length, 5) -- UNSCALED raw OHLCV log returns.
y: (n_samples, 5, 5) -- UNSCALED raw OHLCV log returns (5 days x 5 channels).

RevIN (scaling="std") normalizes per-channel per-sample during forward pass.
Built-in loss compares normalized predictions vs normalized targets.
At inference, prediction_outputs are denormalized by RevIN automatically.
"""

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
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

    RevIN (scaling="std") is kept as default -- handles per-channel per-sample
    normalization internally. DO NOT set scaling=None.

    Args:
        config: Our PatchTSTConfig (num_input_channels=5, prediction_length=5)

    Returns:
        Initialized PatchTSTForPrediction model with RevIN enabled
    """
    hf_config = HFPatchTSTConfig(
        num_input_channels=config.num_input_channels,  # 5 (OHLCV)
        context_length=config.context_length,
        patch_length=config.patch_length,
        stride=config.stride,
        d_model=config.d_model,
        num_attention_heads=config.num_attention_heads,
        num_hidden_layers=config.num_hidden_layers,
        ffn_dim=config.ffn_dim,
        dropout=config.dropout,
        prediction_length=config.prediction_length,  # 5 (direct 5-day)
        # RevIN defaults to scaling="std" -- DO NOT set scaling=None
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
    """Train PatchTST model using PyTorch with multi-task loss on all 5 channels.

    Uses HuggingFace built-in loss by passing future_values=batch_y. The
    built-in loss computes MSE in RevIN-normalized space with equal weight
    on all 5 channels. This is PatchTST's intended multi-task usage.

    Args:
        X: Input sequences, shape (n_samples, context_length, 5) -- UNSCALED OHLCV log returns
        y: Targets, shape (n_samples, 5, 5) -- UNSCALED OHLCV log returns (5 days x 5 channels)
        feature_scaler: Fitted scaler (diagnostic only, not used in training)
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

    print(
        f"[PatchTST] Dataset: {len(X)} samples ({len(X_train)} train, {len(X_val)} val)"
    )
    print(
        f"[PatchTST] Config: {config.epochs} epochs, batch_size={config.batch_size}, lr={config.learning_rate}"
    )
    print(
        f"[PatchTST] Channels: {config.num_input_channels}, context_length={config.context_length}, prediction_length={config.prediction_length}"
    )
    print(
        "[PatchTST] Multi-task: loss on ALL 5 channels, RevIN enabled, targets UNSCALED"
    )

    # Create DataLoaders for memory-efficient batch loading
    # Keep data on CPU and only move batches to GPU on demand
    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Keep simple for compatibility
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    # Create model (RevIN enabled by default)
    model = _create_patchtst_model(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Training loop
    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = 0
    patience_counter = 0

    print("[PatchTST] Starting training...")

    for epoch in range(config.epochs):
        model.train()

        total_train_loss = 0.0
        n_batches = 0
        first_batch_logged = False

        for batch_X, batch_y in train_loader:
            # Move batch to device (memory-efficient: only one batch at a time)
            batch_X = batch_X.to(device)  # (batch, context_length, 5)
            batch_y = batch_y.to(device)  # (batch, 5, 5) = (batch, pred_len, channels)

            optimizer.zero_grad()

            # Use HuggingFace built-in loss by passing future_values
            # Built-in loss computes MSE in RevIN-normalized space on ALL 5 channels
            # This is equal-weight multi-task loss -- PatchTST paper's intended usage
            outputs = model(past_values=batch_X, future_values=batch_y)
            loss = outputs.loss  # MSE on all 5 channels equally in normalized space

            # CRITICAL VERIFICATION: Log model output at epoch 0
            if epoch == 0 and not first_batch_logged:
                first_batch_logged = True
                pred_outputs = outputs.prediction_outputs  # (batch, 5, 5) denormalized
                print("[PatchTST] VERIFY MODEL OUTPUT (epoch 0, batch 0):")
                print(
                    f"  prediction_outputs shape: {pred_outputs.shape} (batch, pred_len=5, channels=5)"
                )
                print(
                    f"  batch_y shape: {batch_y.shape} (batch, pred_len=5, channels=5)"
                )
                print(f"  built-in loss: {loss.item():.6f}")
                print("  First sample, day 0, per-channel predictions (denormalized):")
                for ch_idx, ch_name in enumerate(config.feature_names):
                    pred_val = pred_outputs[0, 0, ch_idx].item()
                    target_val = batch_y[0, 0, ch_idx].item()
                    print(
                        f"    [{ch_idx}] {ch_name}: pred={pred_val:.6f}, target={target_val:.6f}"
                    )

            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.max_grad_norm
            )
            if epoch == 0 and n_batches == 0:
                print("[PatchTST] VERIFY GRADIENTS:")
                print(f"  Gradient norm: {grad_norm:.6f}")
                if grad_norm > 10.0:
                    print(
                        "  WARNING: Large gradient norm (possible exploding gradients)"
                    )
                elif grad_norm < 0.001:
                    print(
                        "  WARNING: Very small gradient norm (possible vanishing gradients)"
                    )

            optimizer.step()

            total_train_loss += loss.item()
            n_batches += 1

            # Explicit cleanup to reduce memory pressure
            del batch_X, batch_y, outputs, loss

        avg_train_loss = total_train_loss / n_batches

        # Validation (batch by batch to save memory)
        # Use built-in loss for consistency with training
        model.eval()
        total_val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_X = val_X.to(device)
                val_y = val_y.to(device)
                val_outputs = model(past_values=val_X, future_values=val_y)
                total_val_loss += val_outputs.loss.item()  # built-in multi-task loss
                n_val_batches += 1
                del val_X, val_y, val_outputs
        val_loss = total_val_loss / n_val_batches

        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Log every epoch
        loss_gap = avg_train_loss - val_loss
        overfitting_indicator = "OVERFITTING" if loss_gap < -0.001 else "OK"
        print(
            f"[PatchTST] Epoch {epoch + 1}/{config.epochs}: "
            f"train_loss={avg_train_loss:.6f}, val_loss={val_loss:.6f}, "
            f"gap={loss_gap:.6f} {overfitting_indicator}, "
            f"lr={current_lr:.6e}, patience={patience_counter}/{config.early_stopping_patience}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(
                    f"[PatchTST] Early stopping triggered at epoch {epoch + 1} "
                    f"(val_loss didn't improve for {config.early_stopping_patience} epochs)"
                )
                break

    # Clear GPU cache to free memory after training loop
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    print(
        f"[PatchTST] Best model at epoch {best_epoch} with val_loss={best_val_loss:.6f}"
    )

    # Restore best model on CPU
    model_cpu = _create_patchtst_model(config)
    if best_model_state is not None:
        model_cpu.load_state_dict(best_model_state)

    # Final metrics (batch by batch to save memory)
    model.eval()
    total_final_train_loss = 0.0
    n_final_batches = 0
    with torch.no_grad():
        for train_X, train_y in train_loader:
            train_X = train_X.to(device)
            train_y = train_y.to(device)
            train_outputs = model(past_values=train_X, future_values=train_y)
            total_final_train_loss += train_outputs.loss.item()
            n_final_batches += 1
            del train_X, train_y, train_outputs
    final_train_loss = total_final_train_loss / n_final_batches

    # Baseline: predict mean return per channel per day
    # y_val shape: (n, 5, 5) -- compute mean across samples for each (day, channel)
    y_val_mean = np.mean(y_val, axis=0, keepdims=True)  # (1, 5, 5)
    baseline_loss = float(np.mean((y_val - y_val_mean) ** 2))

    print(
        f"[PatchTST] Training complete: train_loss={final_train_loss:.6f}, val_loss={best_val_loss:.6f}, baseline={baseline_loss:.6f}"
    )
    beats_baseline = best_val_loss < baseline_loss
    print(f"[PatchTST] Model {'BEATS' if beats_baseline else 'does NOT beat'} baseline")

    # Final GPU cache cleanup before returning
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    return TrainingResult(
        model=model_cpu,
        feature_scaler=feature_scaler,
        config=config,
        train_loss=final_train_loss,
        val_loss=best_val_loss,
        baseline_loss=baseline_loss,
    )

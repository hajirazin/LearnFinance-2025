"""PatchTST model training."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
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

    # Create DataLoaders for memory-efficient batch loading
    # Keep data on CPU and only move batches to GPU on demand
    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float()
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).float()
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Keep simple for compatibility
        pin_memory=True if device.type == 'cuda' else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False,
    )

    # Find close_ret channel index
    try:
        close_ret_idx = config.feature_names.index("close_ret")
    except ValueError:
        raise ValueError(f"close_ret not found in feature_names: {config.feature_names}")

    # Create model
    model = _create_patchtst_model(config).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop
    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = 0
    patience_counter = 0
    log_interval = max(1, config.epochs // 5)

    print("[PatchTST] Starting training...")

    for epoch in range(config.epochs):
        model.train()

        total_train_loss = 0.0
        n_batches = 0
        first_batch_logged = False

        for batch_X, batch_y in train_loader:
            # Move batch to device (memory-efficient: only one batch at a time)
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            # PatchTST outputs prediction_outputs of shape (batch, pred_len=1, channels)
            # We use only close_ret channel for next-day return prediction
            model_outputs = model(past_values=batch_X).prediction_outputs
            # Extract close_ret channel only
            outputs = model_outputs[:, 0, close_ret_idx:close_ret_idx+1]  # Shape: (batch, 1)

            # CRITICAL VERIFICATION: Log model output shape and per-channel predictions
            if epoch == 0 and not first_batch_logged:
                first_batch_logged = True
                print(f"[PatchTST] VERIFY MODEL OUTPUT:")
                print(f"  Full model_outputs shape: {model_outputs.shape} (batch, pred_len, channels)")
                print(f"  Extracted outputs shape: {outputs.shape} (batch, 1) - using close_ret channel only")
                print(f"  First sample per-channel predictions:")
                # Reuse model_outputs we already computed - no extra forward pass
                for ch_idx, ch_name in enumerate(config.feature_names):
                    pred_val = model_outputs[0, 0, ch_idx].item()
                    is_target_channel = "← TARGET" if ch_idx == close_ret_idx else ""
                    print(f"    [{ch_idx}] {ch_name}: {pred_val:.6f} {is_target_channel}")
                print(f"  Batch target (y): {batch_y[0].item():.6f}")
                print(f"  Batch prediction (close_ret): {outputs[0, 0].item():.6f}")

            loss = criterion(outputs, batch_y)
            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            if epoch == 0 and n_batches == 0:
                print(f"[PatchTST] VERIFY GRADIENTS:")
                print(f"  Gradient norm: {grad_norm:.6f}")
                if grad_norm > 10.0:
                    print(f"  WARNING: Large gradient norm (possible exploding gradients)")
                elif grad_norm < 0.001:
                    print(f"  WARNING: Very small gradient norm (possible vanishing gradients)")

            optimizer.step()

            total_train_loss += loss.item()
            n_batches += 1
            
            # Explicit cleanup to reduce memory pressure
            del batch_X, batch_y, model_outputs, outputs, loss

        avg_train_loss = total_train_loss / n_batches

        # Validation (batch by batch to save memory)
        model.eval()
        total_val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_X = val_X.to(device)
                val_y = val_y.to(device)
                val_outputs = model(past_values=val_X).prediction_outputs
                val_outputs = val_outputs[:, 0, close_ret_idx:close_ret_idx+1]
                total_val_loss += criterion(val_outputs, val_y).item()
                n_val_batches += 1
                del val_X, val_y, val_outputs
        val_loss = total_val_loss / n_val_batches

        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # CRITICAL VERIFICATION: Log EVERY epoch to detect overfitting patterns
        loss_gap = avg_train_loss - val_loss
        overfitting_indicator = "⚠️ OVERFITTING" if loss_gap < -0.001 else "✓ OK"
        print(f"[PatchTST] Epoch {epoch + 1}/{config.epochs}: "
              f"train_loss={avg_train_loss:.6f}, val_loss={val_loss:.6f}, "
              f"gap={loss_gap:.6f} {overfitting_indicator}, "
              f"lr={current_lr:.6e}, patience={patience_counter}/{config.early_stopping_patience}")

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
                print(f"[PatchTST] Early stopping triggered at epoch {epoch + 1} "
                      f"(val_loss didn't improve for {config.early_stopping_patience} epochs)")
                break

    # Clear GPU cache to free memory after training loop
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()

    print(f"[PatchTST] Best model at epoch {best_epoch} with val_loss={best_val_loss:.6f}")

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
            train_outputs = model(past_values=train_X).prediction_outputs
            train_outputs = train_outputs[:, 0, close_ret_idx:close_ret_idx+1]
            total_final_train_loss += criterion(train_outputs, train_y).item()
            n_final_batches += 1
            del train_X, train_y, train_outputs
    final_train_loss = total_final_train_loss / n_final_batches

    # Baseline: predict mean return (better than predicting 0)
    y_val_mean = float(np.mean(y_val))
    baseline_loss = float(np.mean((y_val - y_val_mean)**2))

    print(f"[PatchTST] Training complete: train_loss={final_train_loss:.6f}, val_loss={best_val_loss:.6f}, baseline={baseline_loss:.6f}")
    beats_baseline = best_val_loss < baseline_loss
    print(f"[PatchTST] Model {'BEATS' if beats_baseline else 'does NOT beat'} baseline")

    # Final GPU cache cleanup before returning
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()

    return TrainingResult(
        model=model_cpu,
        feature_scaler=feature_scaler,
        config=config,
        train_loss=final_train_loss,
        val_loss=best_val_loss,
        baseline_loss=baseline_loss,
    )


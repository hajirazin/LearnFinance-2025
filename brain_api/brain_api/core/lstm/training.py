"""LSTM model training."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from brain_api.core.lstm.config import LSTMConfig
from brain_api.core.lstm.model import LSTMModel
from brain_api.core.training_utils import get_device


@dataclass
class TrainingResult:
    """Result of LSTM training for next-day return prediction."""

    model: LSTMModel
    feature_scaler: StandardScaler
    config: LSTMConfig
    train_loss: float
    val_loss: float
    baseline_loss: float  # Baseline: predict 0 return (no change)


def train_model_pytorch(
    X: np.ndarray,
    y: np.ndarray,
    feature_scaler: StandardScaler,
    config: LSTMConfig,
) -> TrainingResult:
    """Train LSTM model using PyTorch.

    Automatically uses the best available device:
    - MPS (Apple Silicon GPU) on M1/M2/M3 Macs
    - CUDA (NVIDIA GPU) if available
    - CPU as fallback

    Uses DataLoader for memory-efficient batch loading - keeps data on CPU
    and only moves batches to device on demand.

    Args:
        X: Input sequences, shape (n_samples, seq_len, n_features)
        y: Targets, shape (n_samples, 1) - next-day returns (unscaled)
        feature_scaler: Fitted scaler for input features
        config: Model configuration

    Returns:
        TrainingResult with trained model and metrics
    """
    # Detect best available device
    device = get_device()
    print(f"[LSTM] Training on device: {device}")

    if len(X) == 0:
        print("[LSTM] No training data - returning dummy model")
        model = LSTMModel(config)
        return TrainingResult(
            model=model,
            feature_scaler=feature_scaler,
            config=config,
            train_loss=float("inf"),
            val_loss=float("inf"),
            baseline_loss=float("inf"),
        )

    # Train/val split (time-based, not random, to prevent data leakage)
    split_idx = int(len(X) * (1 - config.validation_split))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"[LSTM] Dataset: {len(X)} samples ({len(X_train)} train, {len(X_val)} val)")
    print(
        f"[LSTM] Config: {config.epochs} epochs, batch_size={config.batch_size}, lr={config.learning_rate}"
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

    # Create model and move to device
    model = LSTMModel(config).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training loop
    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = 0
    log_interval = max(1, config.epochs // 5)  # Log ~5 times during training

    print("[LSTM] Starting training...")

    for epoch in range(config.epochs):
        model.train()

        total_train_loss = 0.0
        n_batches = 0

        for batch_X, batch_y in train_loader:
            # Move batch to device (memory-efficient: only one batch at a time)
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            n_batches += 1

            # Explicit cleanup to reduce memory pressure
            del batch_X, batch_y, outputs, loss

        avg_train_loss = total_train_loss / n_batches

        # Validation (batch by batch to save memory)
        model.eval()
        total_val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_X = val_X.to(device)
                val_y = val_y.to(device)
                val_outputs = model(val_X)
                total_val_loss += criterion(val_outputs, val_y).item()
                n_val_batches += 1
                del val_X, val_y, val_outputs
        val_loss = total_val_loss / n_val_batches

        # Log progress at intervals
        if (epoch + 1) % log_interval == 0 or epoch == 0:
            print(
                f"[LSTM] Epoch {epoch + 1}/{config.epochs}: train_loss={avg_train_loss:.6f}, val_loss={val_loss:.6f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }

        # MPS synchronization and periodic cache clearing to prevent hangs
        # MPS operations are async - sync ensures all ops complete before next epoch
        if device.type == "mps":
            torch.mps.synchronize()
            # Clear cache every 5 epochs to prevent memory buildup
            if (epoch + 1) % 5 == 0:
                torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.synchronize()
            if (epoch + 1) % 5 == 0:
                torch.cuda.empty_cache()

    # Clear GPU cache to free memory after training loop
    if device.type == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    print(f"[LSTM] Best model at epoch {best_epoch} with val_loss={best_val_loss:.6f}")

    # Restore best model (on CPU for portability when saving)
    model_cpu = LSTMModel(config)
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
            train_outputs = model(train_X)
            total_final_train_loss += criterion(train_outputs, train_y).item()
            n_final_batches += 1
            del train_X, train_y, train_outputs
    final_train_loss = total_final_train_loss / n_final_batches

    # Baseline: predict 0 return (no change from day to day)
    # This is the naive "persistence" baseline for returns
    baseline_loss = float(np.mean(y_val**2))

    print(
        f"[LSTM] Training complete: train_loss={final_train_loss:.6f}, val_loss={best_val_loss:.6f}, baseline={baseline_loss:.6f}"
    )
    beats_baseline = best_val_loss < baseline_loss
    print(f"[LSTM] Model {'BEATS' if beats_baseline else 'does NOT beat'} baseline")

    # Final GPU cache cleanup before returning
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    return TrainingResult(
        model=model_cpu,  # Return CPU model for portable saving
        feature_scaler=feature_scaler,
        config=config,
        train_loss=final_train_loss,
        val_loss=best_val_loss,
        baseline_loss=baseline_loss,
    )

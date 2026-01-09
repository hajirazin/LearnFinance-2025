"""LSTM model training."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from brain_api.core.lstm.config import LSTMConfig
from brain_api.core.lstm.model import LSTMModel
from brain_api.core.training_utils import get_device


@dataclass
class TrainingResult:
    """Result of LSTM training for weekly return prediction."""

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

    Args:
        X: Input sequences, shape (n_samples, seq_len, n_features)
        y: Targets, shape (n_samples, 1) - weekly returns (unscaled)
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

    # Convert to tensors and move to device
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)

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

        # Mini-batch training
        indices = torch.randperm(len(X_train_t), device=device)
        total_train_loss = 0.0
        n_batches = 0

        for i in range(0, len(indices), config.batch_size):
            batch_idx = indices[i : i + config.batch_size]
            batch_X = X_train_t[batch_idx]
            batch_y = y_train_t[batch_idx]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_train_loss / n_batches

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()

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

    print(f"[LSTM] Best model at epoch {best_epoch} with val_loss={best_val_loss:.6f}")

    # Restore best model (on CPU for portability when saving)
    model_cpu = LSTMModel(config)
    if best_model_state is not None:
        model_cpu.load_state_dict(best_model_state)

    # Final metrics (compute on device for speed)
    model.eval()
    with torch.no_grad():
        train_outputs = model(X_train_t)
        final_train_loss = criterion(train_outputs, y_train_t).item()

    # Baseline: predict 0 return (no change from week to week)
    # This is the naive "persistence" baseline for returns
    baseline_loss = float(np.mean(y_val**2))

    print(
        f"[LSTM] Training complete: train_loss={final_train_loss:.6f}, val_loss={best_val_loss:.6f}, baseline={baseline_loss:.6f}"
    )
    beats_baseline = best_val_loss < baseline_loss
    print(f"[LSTM] Model {'BEATS' if beats_baseline else 'does NOT beat'} baseline")

    return TrainingResult(
        model=model_cpu,  # Return CPU model for portable saving
        feature_scaler=feature_scaler,
        config=config,
        train_loss=final_train_loss,
        val_loss=best_val_loss,
        baseline_loss=baseline_loss,
    )

"""PatchTST model training.

5-channel OHLCV input with **close-channel-only** training loss.

RevIN (scaling="std") still normalizes inputs per-channel per-sample inside
HuggingFace PatchTST. We do **not** use ``outputs.loss`` (equal-weight MSE on
all denormalized OHLCV channels — volume-dominated). Instead we optimize:

    MSE(prediction_outputs[:, :, close_idx], batch_y[:, :, close_idx])

which matches Alpha-HRP / score-batch ranking on compounded close returns.
OHLCV channels remain in the input; only the loss is close-only.
"""

import threading
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from transformers import PatchTSTForPrediction

from brain_api.core.patchtst.config import PatchTSTConfig
from brain_api.core.training_utils import TrainingCancelledError, get_device


@dataclass
class TrainingResult:
    """Result of PatchTST training."""

    model: PatchTSTForPrediction
    feature_scaler: StandardScaler
    config: PatchTSTConfig
    train_loss: float
    val_loss: float
    baseline_loss: float
    best_epoch: int  # 1-indexed checkpoint restored; 0 if none
    stopped_epoch: int  # 1-indexed last epoch actually run; 0 if none


def _create_patchtst_model(config: PatchTSTConfig) -> PatchTSTForPrediction:
    """Create a HuggingFace PatchTST model from our config.

    RevIN (scaling="std") is kept as default -- handles per-channel per-sample
    normalization internally. DO NOT set scaling=None.

    Args:
        config: Our PatchTSTConfig (num_input_channels=5, prediction_length=5)

    Returns:
        Initialized PatchTSTForPrediction model with RevIN enabled
    """
    return PatchTSTForPrediction(config.to_hf_config())


def _close_channel_index(config: PatchTSTConfig) -> int:
    return config.feature_names.index("close_ret")


def _close_mse(
    preds: torch.Tensor, targets: torch.Tensor, close_idx: int
) -> torch.Tensor:
    """MSE on denormalized close_ret channel only (batch, horizon)."""
    return F.mse_loss(preds[:, :, close_idx], targets[:, :, close_idx])


def _chrono_train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    anchor_dates: np.ndarray | None,
    validation_split: float,
    horizon_purge_calendar_days: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sort by anchor date (if provided) and split train/earlier vs val/later.

    Purges train samples whose target window may overlap the first val anchor
    (conservative calendar-day purge of ``horizon_purge_calendar_days``).
    """
    if anchor_dates is not None:
        if len(anchor_dates) != len(X):
            raise ValueError(
                f"anchor_dates length {len(anchor_dates)} != X length {len(X)}"
            )
        order = np.argsort(anchor_dates)
        X = X[order]
        y = y[order]
        anchor_dates = anchor_dates[order]

    split_idx = int(len(X) * (1 - validation_split))
    if split_idx <= 0 or split_idx >= len(X):
        raise ValueError(
            f"Invalid train/val split_idx={split_idx} for n_samples={len(X)}"
        )

    if anchor_dates is None:
        return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

    min_val_anchor: date = anchor_dates[split_idx]
    purge_before = min_val_anchor - timedelta(days=horizon_purge_calendar_days)
    train_idx = [
        i
        for i in range(split_idx)
        if anchor_dates[i] < purge_before  # strict: no overlap with val horizon
    ]
    if not train_idx:
        # Degenerate tiny panels: fall back to unpurged chronological split
        train_idx = list(range(split_idx))

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]
    return X_train, X_val, y_train, y_val


def train_model_pytorch(
    X: np.ndarray,
    y: np.ndarray,
    feature_scaler: StandardScaler,
    config: PatchTSTConfig,
    shutdown_event: threading.Event | None = None,
    anchor_dates: np.ndarray | None = None,
) -> TrainingResult:
    """Train PatchTST with close-channel-only MSE (denormalized outputs).

    Args:
        X: Input sequences, shape (n_samples, context_length, 5) -- UNSCALED OHLCV
        y: Targets, shape (n_samples, 5, 5) -- UNSCALED OHLCV (5 days x 5 channels)
        feature_scaler: Fitted scaler (diagnostic only, not used in training)
        config: Model configuration
        shutdown_event: Optional cancellation event
        anchor_dates: Optional per-sample anchor dates for chronological split

    Returns:
        TrainingResult with best checkpoint and close-only metrics
    """
    device = get_device()
    print(f"[PatchTST] Training on device: {device}")
    close_idx = _close_channel_index(config)

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
            best_epoch=0,
            stopped_epoch=0,
        )

    X_train, X_val, y_train, y_val = _chrono_train_val_split(
        X, y, anchor_dates, config.validation_split
    )

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
        f"[PatchTST] Loss: close-channel-only MSE (idx={close_idx}), "
        "RevIN on inputs, do not use HF multi-task outputs.loss"
    )

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
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = _create_patchtst_model(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = 0
    stopped_epoch = 0
    patience_counter = 0

    print("[PatchTST] Starting training...")

    for epoch in range(config.epochs):
        if shutdown_event and shutdown_event.is_set():
            print(f"[PatchTST] Training cancelled at epoch {epoch}/{config.epochs}")
            raise TrainingCancelledError("PatchTST training cancelled by shutdown")
        model.train()

        total_train_loss = 0.0
        n_batches = 0
        first_batch_logged = False

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            # Close-only loss on denormalized prediction_outputs (not outputs.loss)
            outputs = model(past_values=batch_X)
            pred_outputs = outputs.prediction_outputs
            loss = _close_mse(pred_outputs, batch_y, close_idx)

            if epoch == 0 and not first_batch_logged:
                first_batch_logged = True
                print("[PatchTST] VERIFY MODEL OUTPUT (epoch 0, batch 0):")
                print(
                    f"  prediction_outputs shape: {pred_outputs.shape} (batch, pred_len=5, channels=5)"
                )
                print(
                    f"  batch_y shape: {batch_y.shape} (batch, pred_len=5, channels=5)"
                )
                print(f"  close-only loss: {loss.item():.6f}")
                for ch_idx, ch_name in enumerate(config.feature_names):
                    pred_val = pred_outputs[0, 0, ch_idx].item()
                    target_val = batch_y[0, 0, ch_idx].item()
                    print(
                        f"    [{ch_idx}] {ch_name}: pred={pred_val:.6f}, target={target_val:.6f}"
                    )

            loss.backward()

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

            del batch_X, batch_y, outputs, pred_outputs, loss

        avg_train_loss = total_train_loss / n_batches

        model.eval()
        total_val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_X = val_X.to(device)
                val_y = val_y.to(device)
                preds = model(past_values=val_X).prediction_outputs
                total_val_loss += _close_mse(preds, val_y, close_idx).item()
                n_val_batches += 1
                del val_X, val_y, preds
        val_loss = total_val_loss / n_val_batches
        stopped_epoch = epoch + 1

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

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

    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    print(
        f"[PatchTST] Best model at epoch {best_epoch} with "
        f"val_loss={best_val_loss:.6f} (close-only)"
    )

    # Restore best weights onto the training device model for final metrics
    # (Phase B: metrics must match the promoted checkpoint, not last epoch).
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model_cpu = _create_patchtst_model(config)
    if best_model_state is not None:
        model_cpu.load_state_dict(best_model_state)

    model.eval()

    total_final_train_loss = 0.0
    n_final_batches = 0
    with torch.no_grad():
        for train_X, train_y in train_loader:
            train_X = train_X.to(device)
            train_y = train_y.to(device)
            preds = model(past_values=train_X).prediction_outputs
            total_final_train_loss += _close_mse(preds, train_y, close_idx).item()
            n_final_batches += 1
            del train_X, train_y, preds
    final_train_loss = total_final_train_loss / n_final_batches

    total_raw_val_loss = 0.0
    n_raw_val_batches = 0
    with torch.no_grad():
        for val_X, val_y in val_loader:
            val_X = val_X.to(device)
            val_y = val_y.to(device)
            preds = model(past_values=val_X).prediction_outputs
            total_raw_val_loss += _close_mse(preds, val_y, close_idx).item()
            n_raw_val_batches += 1
            del val_X, val_y, preds
    final_val_loss = total_raw_val_loss / n_raw_val_batches

    # Baseline: predict mean close return per day on val (close channel only)
    y_val_close = y_val[:, :, close_idx]
    y_val_close_mean = np.mean(y_val_close, axis=0, keepdims=True)
    baseline_loss = float(np.mean((y_val_close - y_val_close_mean) ** 2))

    print(
        f"[PatchTST] Training complete (close-only): "
        f"train_loss={final_train_loss:.6f}, val_loss={final_val_loss:.6f}, "
        f"baseline={baseline_loss:.6f}"
    )
    beats_baseline = final_val_loss < baseline_loss
    print(
        f"[PatchTST] Model {'BEATS' if beats_baseline else 'does NOT beat'} "
        "close-only baseline"
    )

    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    return TrainingResult(
        model=model_cpu,
        feature_scaler=feature_scaler,
        config=config,
        train_loss=final_train_loss,
        val_loss=final_val_loss,
        baseline_loss=baseline_loss,
        best_epoch=best_epoch,
        stopped_epoch=stopped_epoch,
    )

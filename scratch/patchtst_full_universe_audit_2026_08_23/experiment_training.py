"""Deterministic training and prediction for the two frozen experiment arms."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from experiment_metrics import aggregate_metrics
from experiment_spec import (
    CLOSE_INDEX,
    ExperimentArm,
    build_model,
    json_dump,
    set_seed,
    sha256_file,
)
from torch.utils.data import DataLoader, TensorDataset


def fit_target_scales(train_y: np.ndarray) -> np.ndarray:
    """Fit per-channel target standard deviations on training labels only."""
    if train_y.ndim != 3 or train_y.shape[-1] != 5:
        raise ValueError("train_y must have shape (samples, horizon, 5)")
    scales = train_y.reshape(-1, 5).std(axis=0, ddof=0).astype(np.float32)
    scales[~np.isfinite(scales) | (scales < 1e-8)] = 1.0
    return scales


def scaled_channel_mse(
    predictions: torch.Tensor, targets: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    """Equal-channel MSE after division by train-only target scale."""
    if scales.shape != (predictions.shape[-1],):
        raise ValueError("target scales do not match prediction channels")
    return torch.mean(((predictions - targets) / scales.view(1, 1, -1)) ** 2)


def _objective(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    arm: ExperimentArm,
    scales: torch.Tensor,
) -> torch.Tensor:
    if arm.objective == "close_daily_mse":
        return F.mse_loss(predictions[:, :, CLOSE_INDEX], targets[:, :, CLOSE_INDEX])
    if arm.objective == "scaled_ohlcv_daily_mse":
        return scaled_channel_mse(predictions, targets, scales)
    raise ValueError(f"unknown objective {arm.objective}")


def predict_log_returns(
    model: torch.nn.Module,
    x: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size]).to(device)
            predictions = model(past_values=batch).prediction_outputs
            outputs.append(predictions[:, :, CLOSE_INDEX].sum(dim=1).cpu().numpy())
    result = np.concatenate(outputs).astype(np.float64)
    if not np.isfinite(result).all():
        raise FloatingPointError("model produced nonfinite weekly predictions")
    return result


def prediction_frame(
    metadata: pd.DataFrame, predicted_log_returns: np.ndarray
) -> pd.DataFrame:
    if len(metadata) != len(predicted_log_returns):
        raise ValueError("prediction length does not match metadata")
    frame = metadata[["decision_date", "symbol", "actual_weekly_log_return"]].copy()
    frame["actual_weekly_return"] = np.expm1(
        frame.pop("actual_weekly_log_return").to_numpy(float)
    )
    frame["predicted_weekly_return"] = np.expm1(predicted_log_returns)
    return frame


def _validation_loss(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    arm: ExperimentArm,
    scales: torch.Tensor,
    device: torch.device,
) -> float:
    model.eval()
    weighted = 0.0
    count = 0
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=1024,
        shuffle=False,
    )
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            predictions = model(past_values=batch_x).prediction_outputs
            loss = _objective(predictions, batch_y, arm, scales)
            weighted += float(loss) * len(batch_x)
            count += len(batch_x)
    return weighted / count


def _fingerprint(
    arm: ExperimentArm,
    seed: int,
    arrays: tuple[np.ndarray, ...],
    training_config: dict[str, Any],
) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(asdict(arm), sort_keys=True).encode())
    digest.update(str(seed).encode())
    digest.update(json.dumps(training_config, sort_keys=True).encode())
    digest.update(Path(__file__).read_bytes())
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def train_arm(
    arm: ExperimentArm,
    seed: int,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    validation_metadata: pd.DataFrame,
    *,
    model_dir: Path,
    device: torch.device,
    max_epochs: int = 60,
    patience: int = 8,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Train one seed and restore the checkpoint with best validation rank IC."""
    training_config = {
        "max_epochs": max_epochs,
        "patience": patience,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": "Adam",
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "scheduler": "ReduceLROnPlateau(factor=0.5,patience=5)",
        "selection": "max validation weekly rank IC; validation loss tie-break",
    }
    fingerprint = _fingerprint(
        arm,
        seed,
        (train_x, train_y, validation_x, validation_y),
        training_config,
    )
    weights_path = model_dir / "weights.pt"
    metadata_path = model_dir / "metadata.json"
    if weights_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("fingerprint") == fingerprint and metadata.get(
            "weights_sha256"
        ) == sha256_file(weights_path):
            model = build_model(arm)
            model.load_state_dict(
                torch.load(weights_path, map_location="cpu", weights_only=True)
            )
            return model, metadata

    set_seed(seed)
    model = build_model(arm).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    scales_np = fit_target_scales(train_y)
    scales = torch.from_numpy(scales_np).to(device)
    dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    best_rank_ic = -math.inf
    best_val_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    start_time = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        generator = torch.Generator().manual_seed(seed + epoch)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        train_weighted = 0.0
        train_count = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(past_values=batch_x).prediction_outputs
            loss = _objective(predictions, batch_y, arm, scales)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"nonfinite training loss for {arm.name}/{seed}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_weighted += float(loss.detach()) * len(batch_x)
            train_count += len(batch_x)

        val_loss = _validation_loss(
            model, validation_x, validation_y, arm, scales, device
        )
        validation_predictions = predict_log_returns(model, validation_x, device=device)
        validation_frame = prediction_frame(validation_metadata, validation_predictions)
        val_rank_ic = float(aggregate_metrics(validation_frame)["weekly_rank_ic"])
        scheduler.step(val_loss)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_weighted / train_count,
                "validation_loss": val_loss,
                "validation_rank_ic": val_rank_ic,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        improved = val_rank_ic > best_rank_ic + 1e-12 or (
            math.isclose(val_rank_ic, best_rank_ic, abs_tol=1e-12)
            and val_loss < best_val_loss
        )
        if improved:
            best_rank_ic = val_rank_ic
            best_val_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError(f"no finite checkpoint for {arm.name}/{seed}")
    restored = build_model(arm)
    restored.load_state_dict(best_state)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, weights_path)
    metadata = {
        "arm": asdict(arm),
        "seed": seed,
        "fingerprint": fingerprint,
        "weights_sha256": sha256_file(weights_path),
        "target_scales": scales_np.tolist(),
        "best_epoch": best_epoch,
        "stopped_epoch": history[-1]["epoch"],
        "best_validation_rank_ic": best_rank_ic,
        "best_validation_loss": best_val_loss,
        "runtime_seconds": time.perf_counter() - start_time,
        "n_train": len(train_x),
        "n_validation": len(validation_x),
        "training_config": training_config,
        "history": history,
    }
    json_dump(metadata_path, metadata)
    return restored, metadata

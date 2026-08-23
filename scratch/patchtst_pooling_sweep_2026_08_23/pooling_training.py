"""Exclusive deterministic training for one pooling-head/fold/seed at a time."""

from __future__ import annotations

import fcntl
import gc
import hashlib
import json
import math
import time
from contextlib import AbstractContextManager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pooling_spec import (
    EVALUATION_FOLDS,
    POOLING_HEADS,
    SEEDS,
    EvaluationFold,
    PoolingHead,
    build_patchtst_model,
    json_dump,
    set_deterministic_seed,
    sha256_file,
)
from torch.utils.data import DataLoader, TensorDataset


class TrainingJob(NamedTuple):
    fold_name: str
    pooling_name: str
    seed: int


def training_jobs() -> tuple[TrainingJob, ...]:
    """Return the only permitted serial execution order."""
    return tuple(
        TrainingJob(fold_name, pooling_name, seed)
        for fold_name in EVALUATION_FOLDS
        for pooling_name in POOLING_HEADS
        for seed in SEEDS
    )


class TrainingRunLock(AbstractContextManager["TrainingRunLock"]):
    """Prevent a second heavy research runner from sharing this experiment."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: Any = None

    def __enter__(self) -> TrainingRunLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+")
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._handle.close()
            self._handle = None
            raise RuntimeError("a pooling training runner is already active") from error
        self._handle.seek(0)
        self._handle.truncate()
        self._handle.write(f"pid={__import__('os').getpid()}\n")
        self._handle.flush()
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None


def training_fingerprint(
    fold: EvaluationFold,
    head: PoolingHead,
    seed: int,
    arrays: tuple[np.ndarray, ...],
    *,
    max_epochs: int,
    patience: int,
) -> str:
    """Hash every input that can change a trained checkpoint."""
    digest = hashlib.sha256()
    digest.update(json.dumps(asdict(fold), sort_keys=True, default=str).encode())
    digest.update(json.dumps(asdict(head), sort_keys=True).encode())
    digest.update(str(seed).encode())
    digest.update(str(max_epochs).encode())
    digest.update(str(patience).encode())
    digest.update(Path(__file__).read_bytes())
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode())
        digest.update(str(contiguous.dtype).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _mean_weekly_rank_ic(metadata: pd.DataFrame, predicted_logs: np.ndarray) -> float:
    frame = metadata[["decision_date", "symbol", "actual_weekly_log_return"]].copy()
    frame["predicted"] = predicted_logs
    values: list[float] = []
    for _, week in frame.groupby("decision_date", sort=True):
        if len(week) < 3 or week["predicted"].std(ddof=0) < 1e-12:
            continue
        values.append(
            float(
                week["predicted"]
                .rank(method="average")
                .corr(week["actual_weekly_log_return"].rank(method="average"))
            )
        )
    if not values or not np.isfinite(values).all():
        raise FloatingPointError("validation weekly rank IC is not finite")
    return float(np.mean(values))


def predict_weekly_log_returns(
    model: torch.nn.Module,
    x: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 1_024,
) -> np.ndarray:
    """Predict compounded weekly log returns using blocking device transfers."""
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size]).to(device)
            prediction = model(past_values=batch).prediction_outputs[:, :, 0]
            outputs.append(prediction.sum(dim=1).cpu().numpy())
    result = np.concatenate(outputs).astype(np.float64)
    if not np.isfinite(result).all():
        raise FloatingPointError("model produced nonfinite weekly predictions")
    return result


def _validation_loss(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    *,
    device: torch.device,
) -> float:
    model.eval()
    weighted = 0.0
    count = 0
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=1_024,
        shuffle=False,
    )
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            prediction = model(past_values=batch_x).prediction_outputs
            loss = F.mse_loss(prediction[:, :, 0], batch_y[:, :, 0])
            weighted += float(loss) * len(batch_x)
            count += len(batch_x)
    return weighted / count


def train_pooling_seed(
    fold: EvaluationFold,
    head: PoolingHead,
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
    """Train and restore one checkpoint selected by validation rank IC."""
    arrays = (train_x, train_y, validation_x, validation_y)
    fingerprint = training_fingerprint(
        fold,
        head,
        seed,
        arrays,
        max_epochs=max_epochs,
        patience=patience,
    )
    weights_path = model_dir / "weights.pt"
    metadata_path = model_dir / "metadata.json"
    if weights_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("fingerprint") == fingerprint and metadata.get(
            "weights_sha256"
        ) == sha256_file(weights_path):
            restored = build_patchtst_model(head)
            restored.load_state_dict(
                torch.load(weights_path, map_location="cpu", weights_only=True)
            )
            return restored, metadata

    set_deterministic_seed(seed)
    model = build_patchtst_model(head).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    best_rank_ic = -math.inf
    best_validation_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()

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
            prediction = model(past_values=batch_x).prediction_outputs
            loss = F.mse_loss(prediction[:, :, 0], batch_y[:, :, 0])
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"nonfinite loss for {fold.name}/{head.name}/{seed}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_weighted += float(loss.detach()) * len(batch_x)
            train_count += len(batch_x)

        validation_loss = _validation_loss(
            model, validation_x, validation_y, device=device
        )
        validation_predictions = predict_weekly_log_returns(
            model, validation_x, device=device
        )
        validation_rank_ic = _mean_weekly_rank_ic(
            validation_metadata, validation_predictions
        )
        scheduler.step(validation_loss)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_weighted / train_count,
                "validation_loss": validation_loss,
                "validation_rank_ic": validation_rank_ic,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        improved = validation_rank_ic > best_rank_ic + 1e-12 or (
            math.isclose(validation_rank_ic, best_rank_ic, abs_tol=1e-12)
            and validation_loss < best_validation_loss
        )
        if improved:
            best_rank_ic = validation_rank_ic
            best_validation_loss = validation_loss
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
        raise RuntimeError(f"no checkpoint for {fold.name}/{head.name}/{seed}")
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, weights_path)
    metadata = {
        "fold": asdict(fold),
        "pooling_head": asdict(head),
        "seed": seed,
        "fingerprint": fingerprint,
        "weights_sha256": sha256_file(weights_path),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        "device": str(device),
        "n_train": len(train_x),
        "n_validation": len(validation_x),
        "best_epoch": best_epoch,
        "stopped_epoch": history[-1]["epoch"],
        "best_validation_rank_ic": best_rank_ic,
        "best_validation_loss": best_validation_loss,
        "training_config": {
            "objective": "daily_close_mse",
            "optimizer": "Adam",
            "learning_rate": learning_rate,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "patience": patience,
            "max_grad_norm": 1.0,
            "scheduler": "ReduceLROnPlateau(mode=min,factor=0.5,patience=5)",
        },
        "history": history,
    }
    json_dump(metadata_path, metadata)
    restored = build_patchtst_model(head)
    restored.load_state_dict(best_state)
    return restored, metadata


def cleanup_device(model: torch.nn.Module | None) -> None:
    """Release one completed model before the next sequential job."""
    if model is not None:
        model.cpu()
        del model
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()
    gc.collect()


def _tensor_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = np.ascontiguousarray(value.detach().cpu().numpy())
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def run_mps_determinism_smoke() -> dict[str, Any]:
    """Require two identical one-step MPS trainings before the full sweep."""
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is not available")
    device = torch.device("mps")
    head = POOLING_HEADS["mean"]
    rng = np.random.default_rng(20260823)
    x = rng.normal(0, 0.01, size=(4, 60, 1)).astype(np.float32)
    y = rng.normal(0, 0.01, size=(4, 5, 1)).astype(np.float32)
    runs: list[tuple[str, str]] = []
    started = time.perf_counter()
    for _ in range(2):
        set_deterministic_seed(20260823)
        model = build_patchtst_model(head).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        batch_x = torch.from_numpy(x).to(device)
        batch_y = torch.from_numpy(y).to(device)
        optimizer.zero_grad()
        prediction = model(past_values=batch_x).prediction_outputs
        loss = F.mse_loss(prediction[:, :, 0], batch_y[:, :, 0])
        loss.backward()
        optimizer.step()
        torch.mps.synchronize()
        model.eval()
        with torch.no_grad():
            output = model(past_values=batch_x).prediction_outputs.cpu().numpy()
        runs.append(
            (
                _tensor_state_sha256(model),
                hashlib.sha256(np.ascontiguousarray(output).tobytes()).hexdigest(),
            )
        )
        cleanup_device(model)
    return {
        "passed": runs[0] == runs[1],
        "device": "mps",
        "state_sha256": runs[0][0],
        "prediction_sha256": runs[0][1],
        "repeat_state_sha256": runs[1][0],
        "repeat_prediction_sha256": runs[1][1],
        "runtime_seconds": time.perf_counter() - started,
        "hardware": {
            "chip": "Apple M5 Pro",
            "gpu_cores": 20,
            "unified_memory_gb": 48,
        },
    }


def mps_smoke_result_is_sanitized(result: dict[str, Any]) -> bool:
    """Reject accidental persistence of machine identifiers."""
    forbidden = ("serial", "uuid", "udid")

    def keys(value: Any) -> list[str]:
        if isinstance(value, dict):
            return [str(key).lower() for key in value] + [
                nested for inner in value.values() for nested in keys(inner)
            ]
        if isinstance(value, list):
            return [nested for inner in value for nested in keys(inner)]
        return []

    return not any(token in key for key in keys(result) for token in forbidden)

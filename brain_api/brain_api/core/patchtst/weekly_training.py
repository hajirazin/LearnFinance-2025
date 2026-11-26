"""Fit the direct five-session PatchTST regressor."""

import math
import threading
from dataclasses import dataclass

import numpy as np
import torch

from brain_api.core.patchtst.config import PatchTSTConfig, create_patchtst_model
from brain_api.core.patchtst.weekly_data import (
    InsufficientWeeklyHistory,
    WeeklyPatchTSTData,
)
from brain_api.core.patchtst.weekly_rank_ic import mean_weekly_rank_ic
from brain_api.core.training_utils import TrainingCancelledError, get_device


@dataclass(frozen=True)
class PatchTSTTargetScale:
    target_std: float

    def __post_init__(self):
        if not math.isfinite(self.target_std) or self.target_std <= 0:
            raise ValueError("Weekly PatchTST target scale must be finite and positive")


@dataclass
class WeeklyPatchTSTResult:
    model: torch.nn.Module
    feature_scaler: PatchTSTTargetScale
    train_loss: float
    val_loss: float
    baseline_loss: float
    best_epoch: int
    stopped_epoch: int
    val_rank_ic: float | None
    evidence: dict


def require_target_scale(value) -> PatchTSTTargetScale:
    if not isinstance(value, PatchTSTTargetScale):
        raise ValueError("Weekly PatchTST artifact lacks its fitted target scale")
    return PatchTSTTargetScale(value.target_std)


def weekly_fold_indices(data: WeeklyPatchTSTData, fraction: float):
    anchors = np.unique(data.anchor_dates)
    val_count = max(13, math.ceil(len(anchors) * fraction))
    if len(anchors) <= val_count:
        raise InsufficientWeeklyHistory("Weekly PatchTST has no eligible fitting fold")
    boundary = anchors[-val_count]
    training = np.flatnonzero(
        (data.anchor_dates < boundary) & (data.target_dates < boundary)
    )
    validation = np.flatnonzero(data.anchor_dates >= boundary)
    if len(np.unique(data.anchor_dates[training])) < 52:
        raise InsufficientWeeklyHistory(
            "Weekly PatchTST needs 52 fitting weeks after purging"
        )
    return training, validation


def predict_weekly_returns(
    model, matrix: np.ndarray, scaler, batch_size: int = 256
) -> np.ndarray:
    factor = require_target_scale(scaler).target_std
    model.eval()
    device = next(model.parameters()).device
    with torch.inference_mode():
        chunks = [
            model(
                past_values=torch.from_numpy(matrix[start : start + batch_size]).to(
                    device
                )
            )
            .regression_outputs[:, 0]
            .cpu()
            .numpy()
            for start in range(0, len(matrix), batch_size)
        ]
    output = np.concatenate(chunks) * factor
    if not np.isfinite(output).all():
        raise FloatingPointError("Weekly PatchTST produced nonfinite forecasts")
    return output


def fit_weekly_patchtst(
    data: WeeklyPatchTSTData,
    config: PatchTSTConfig,
    *,
    shutdown_event: threading.Event | None = None,
) -> WeeklyPatchTSTResult:
    train, validation = weekly_fold_indices(data, config.validation_split)
    targets = data.y
    scale = PatchTSTTargetScale(float(np.std(targets[train], dtype=np.float64)))
    validation_dates = data.anchor_dates[validation]
    _, reverse, counts = np.unique(
        validation_dates, return_inverse=True, return_counts=True
    )
    weights = 1.0 / counts[reverse]
    baseline = float(
        np.average((targets[validation] - targets[train].mean()) ** 2, weights=weights)
    )

    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    device = get_device()
    model = create_patchtst_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5
    )
    best = float("inf")
    state = None
    best_epoch = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = rng.permutation(train)
        running_loss = 0.0
        for start in range(0, len(order), config.batch_size):
            if shutdown_event is not None and shutdown_event.is_set():
                raise TrainingCancelledError("Weekly PatchTST training cancelled")
            rows = order[start : start + config.batch_size]
            inputs = torch.from_numpy(data.X[rows]).to(device)
            labels = torch.from_numpy(targets[rows] / scale.target_std).to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(past_values=inputs).regression_outputs[:, 0]
            loss = torch.nn.functional.mse_loss(predictions, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite weekly PatchTST training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            running_loss += float(loss.detach()) * len(rows) * scale.target_std**2

        estimates = np.concatenate(
            [
                predict_weekly_returns(model, data.X[rows], scale, config.batch_size)
                for rows in np.array_split(
                    validation, math.ceil(len(validation) / config.batch_size)
                )
            ]
        )
        mse = float(np.average((estimates - targets[validation]) ** 2, weights=weights))
        scheduler.step(mse)
        if mse < best:
            best, best_epoch = mse, epoch
            best_train = running_loss / len(train)
            best_predictions = estimates
            state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in model.state_dict().items()
            }
        if epoch - best_epoch >= config.early_stopping_patience:
            break
    if state is None:
        raise ValueError("Weekly PatchTST selected no checkpoint")
    model.load_state_dict(state)
    model.eval()
    try:
        rank_ic = mean_weekly_rank_ic(
            validation_dates,
            data.symbols[validation],
            best_predictions,
            targets[validation],
        )
    except FloatingPointError:
        rank_ic = None
    return WeeklyPatchTSTResult(
        model,
        scale,
        best_train,
        best,
        baseline,
        best_epoch,
        epoch,
        rank_ic,
        {
            "weekly_rmse_pp": 100 * math.sqrt(best),
            "latest_label_end": max(data.target_dates).isoformat(),
            "last_usable_session": max(data.target_dates).isoformat(),
            "latest_label_symbols": int(
                np.sum(data.target_dates == max(data.target_dates))
            ),
            "fit_examples": len(train),
            "validation_examples": len(validation),
            "fit_weeks": len(np.unique(data.anchor_dates[train])),
            "validation_weeks": len(counts),
        },
    )

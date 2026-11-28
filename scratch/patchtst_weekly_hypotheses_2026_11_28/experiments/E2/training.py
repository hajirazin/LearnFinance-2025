"""Scratch loss ablation preserving the production weekly MSE fit mechanics."""

import math
import time

import numpy as np
import torch
from evidence import sha256, write_json

from brain_api.core.patchtst.config import create_patchtst_model
from brain_api.core.patchtst.weekly_rank_ic import mean_weekly_rank_ic
from brain_api.core.patchtst.weekly_training import (
    PatchTSTTargetScale,
    predict_weekly_returns,
    weekly_fold_indices,
)

RAW_HUBER_DELTA = 0.05


def training_objective(predicted, actual, arm, target_std, *, reduction="mean"):
    if not math.isfinite(target_std) or target_std <= 0:
        raise ValueError("Target standard deviation must be finite and positive")
    if arm == "mse":
        return torch.nn.functional.mse_loss(predicted, actual, reduction=reduction)
    if arm == "huber":
        return 2 * torch.nn.functional.huber_loss(
            predicted,
            actual,
            delta=RAW_HUBER_DELTA / target_std,
            reduction=reduction,
        )
    raise ValueError(f"Unknown objective: {arm}")


def fit_arm(data, config, arm, device, output):
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    train, validation = weekly_fold_indices(data, config.validation_split)
    targets = data.y
    scale = PatchTSTTargetScale(float(np.std(targets[train], dtype=np.float64)))
    dates = data.anchor_dates[validation]
    _, reverse, counts = np.unique(dates, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[reverse]
    baseline = float(
        np.average((targets[validation] - targets[train].mean()) ** 2, weights=weights)
    )
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    model = create_patchtst_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5
    )
    initial_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(initial_state, output / "initial.pt")
    best, best_epoch, history = float("inf"), 0, []
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = rng.permutation(train)
        running_objective, running_mse, norms = 0.0, 0.0, []
        for start in range(0, len(order), config.batch_size):
            rows = order[start : start + config.batch_size]
            inputs = torch.from_numpy(data.X[rows]).to(device)
            labels = torch.from_numpy(targets[rows] / scale.target_std).to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(past_values=inputs).regression_outputs[:, 0]
            loss = training_objective(predictions, labels, arm, scale.target_std)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite {arm} objective")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.max_grad_norm, error_if_nonfinite=True
            )
            optimizer.step()
            norms.append(float(norm))
            running_objective += float(loss.detach()) * len(rows) * scale.target_std**2
            running_mse += (
                float(torch.nn.functional.mse_loss(predictions.detach(), labels))
                * len(rows)
                * scale.target_std**2
            )
        estimates = np.concatenate(
            [
                predict_weekly_returns(model, data.X[rows], scale, config.batch_size)
                for rows in np.array_split(
                    validation, math.ceil(len(validation) / config.batch_size)
                )
            ]
        )
        mse = float(np.average((estimates - targets[validation]) ** 2, weights=weights))
        if not math.isfinite(mse):
            raise FloatingPointError("Nonfinite validation MSE")
        scheduler.step(mse)
        if mse < best:
            best, best_epoch = mse, epoch
            torch.save(
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                output / "model.pt",
            )
        rank_ic = mean_weekly_rank_ic(
            dates, data.symbols[validation], estimates, targets[validation]
        )
        history.append(
            {
                "epoch": epoch,
                "training_objective_raw_units": running_objective / len(train),
                "training_mse": running_mse / len(train),
                "validation_mse": mse,
                "validation_rank_ic": rank_ic,
                "validation_prediction_std": float(np.std(estimates)),
                "validation_prediction_mean": float(np.mean(estimates)),
                "gradient_norm_mean": float(np.mean(norms)),
                "gradient_clipped_fraction": float(
                    np.mean(np.asarray(norms) > config.max_grad_norm)
                ),
                "best_epoch": best_epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        write_json(output / "progress.json", {"history": history})
        print(
            f"{arm} epoch={epoch} train_mse={running_mse / len(train):.8f} "
            f"val_mse={mse:.8f} val_ic={rank_ic:.5f} best={best_epoch} "
            f"elapsed={time.monotonic() - started:.1f}s",
            flush=True,
        )
        if epoch - best_epoch >= config.early_stopping_patience:
            break
    result = {
        "arm": arm,
        "best_epoch": best_epoch,
        "stopped_epoch": epoch,
        "validation_mse": best,
        "baseline_validation_mse": baseline,
        "target_std": scale.target_std,
        "training_mean": float(targets[train].mean()),
        "raw_huber_delta": RAW_HUBER_DELTA if arm == "huber" else None,
        "history": history,
        "checkpoint_sha256": sha256(output / "model.pt"),
        "device": str(device),
    }
    write_json(output / "result.json", result)
    del model, optimizer, initial_state
    if device.type == "mps":
        torch.mps.empty_cache()
    return result

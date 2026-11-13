"""Fit and score the two PatchTST screens in the Friday replay."""

from dataclasses import replace
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from brain_api.core.patchtst.config import DEFAULT_CONFIG
from brain_api.core.patchtst.weekly_rank_ic import checkpoint_is_better
from sklearn.preprocessing import StandardScaler
from transformers import PatchTSTForPrediction, PatchTSTForRegression

SEED = 42
WEEK_PATCH = 4
WEEK_STRIDE = 2


def simple_returns(output, *, daily):
    """Convert daily log returns or a direct weekly output to one simple return."""
    if daily:
        return torch.expm1(output.sum(dim=1))
    return output.squeeze(-1)


def build_model(x, *, daily):
    channels = x.shape[-1]
    config = replace(
        DEFAULT_CONFIG, context_length=x.shape[1], num_input_channels=channels
    )
    if daily:
        return PatchTSTForPrediction(config.to_hf_config())
    config = replace(
        config, patch_length=WEEK_PATCH, stride=WEEK_STRIDE, prediction_length=1
    )
    hf_config = config.to_hf_config()
    hf_config.scaling = None  # Train-fold scaling preserves absolute feature levels.
    hf_config.num_targets = 1
    return PatchTSTForRegression(hf_config)


def model_output(model, x, observed, *, daily):
    if daily:
        return model(past_values=x, past_observed_mask=observed).prediction_outputs[
            :, :, 0
        ]
    return model(past_values=x).regression_outputs


def fit_patchtst(
    x,
    target,
    observed,
    train,
    validation,
    score,
    labels,
    *,
    daily,
    device,
    epochs,
    batch_size,
    model_factory=build_model,
):
    """Train from scratch and restore the best validation weekly rank-IC checkpoint."""
    started = perf_counter()
    torch.manual_seed(SEED)
    model = model_factory(x, daily=daily).to(device)
    model_name = "daily" if daily else "weekly"
    indices = np.flatnonzero(train | validation | score)
    x = x[indices]
    target = target[indices]
    observed = observed[indices]
    train, validation, score = train[indices], validation[indices], score[indices]
    val_labels = labels.iloc[indices[validation]]
    val_actual = val_labels["forward_return"]
    # Weekly channel standardization and target scaling are fit on training only.
    # Target scaling multiplies squared error by a constant and preserves its optimum.
    if daily:
        target_scale = 1.0
    else:
        scaler = StandardScaler().fit(x[train].reshape(-1, x.shape[-1]))
        x = scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
        target_scale = float(np.std(target[train], dtype=np.float64))
        if not np.isfinite(target_scale) or target_scale <= 0:
            raise ValueError("Training target scale must be positive and finite")
    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    observed = torch.as_tensor(observed, dtype=torch.bool, device=device)
    target = torch.as_tensor(target / target_scale, dtype=torch.float32, device=device)
    train_rows = torch.as_tensor(np.flatnonzero(train), device=device)
    val_rows = torch.as_tensor(np.flatnonzero(validation), device=device)
    score_rows = torch.as_tensor(np.flatnonzero(score), device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_CONFIG.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5
    )
    actual_rank = val_actual.groupby(level="friday").rank().to_numpy()
    groups = val_labels.index.get_level_values("friday").factorize(sort=True)[0]
    week_rows = [np.flatnonzero(groups == week) for week in np.unique(groups)]

    def predictions(rows):
        model.eval()
        with torch.inference_mode():
            return torch.cat(
                [
                    model_output(model, x[ids], observed[ids], daily=daily)
                    for ids in rows.split(batch_size)
                ]
            )

    best_state = None
    best_ic, best_mse = -np.inf, np.inf
    history = []
    generator = torch.Generator().manual_seed(SEED)
    for epoch in range(1, epochs + 1):
        epoch_start = perf_counter()
        model.train()
        order = torch.randperm(len(train_rows), generator=generator).to(device)
        running_loss = torch.zeros((), device=device)
        for rows in train_rows[order].split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            output = model_output(model, x[rows], observed[rows], daily=daily)
            loss = F.mse_loss(output, target[rows])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), DEFAULT_CONFIG.max_grad_norm
            )
            optimizer.step()
            running_loss += loss.detach() * len(rows)
        val_output = predictions(val_rows)
        val_mse = float(F.mse_loss(val_output, target[val_rows])) * target_scale**2
        val_pred = simple_returns(val_output * target_scale, daily=daily).cpu().numpy()
        pred_rank = val_actual.copy()
        pred_rank.iloc[:] = val_pred
        pred_rank = pred_rank.groupby(level="friday").rank().to_numpy()
        correlations = [
            np.corrcoef(pred_rank[rows], actual_rank[rows])[0, 1]
            for rows in week_rows
            if len(rows) >= 3
            and np.ptp(pred_rank[rows]) > 0
            and np.ptp(actual_rank[rows]) > 0
        ]
        val_ic = float(np.mean(correlations)) if correlations else np.nan
        train_mse = float(running_loss) * target_scale**2 / len(train_rows)
        if not np.isfinite([val_ic, val_mse, train_mse]).all():
            raise FloatingPointError(
                f"Non-finite {model_name} PatchTST validation or loss"
            )
        scheduler.step(val_mse)
        history.append(
            {
                "epoch": epoch,
                "train_mse": train_mse,
                "validation_mse": val_mse,
                "validation_rank_ic": val_ic,
            }
        )
        if checkpoint_is_better(val_ic, val_mse, best_ic, best_mse):
            best_ic, best_mse, best_epoch = val_ic, val_mse, epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        print(
            f"  {model_name} epoch {epoch}: val IC {val_ic:+.4f}, {perf_counter() - epoch_start:.1f}s",
            flush=True,
        )
        if epoch - best_epoch >= DEFAULT_CONFIG.early_stopping_patience:
            break

    model.load_state_dict(best_state)
    predicted = (
        simple_returns(predictions(score_rows) * target_scale, daily=daily)
        .cpu()
        .numpy()
    )
    if not np.isfinite(predicted).all():
        raise FloatingPointError(f"Non-finite {model_name} PatchTST forecasts")
    metadata = {
        "config": model.config.to_dict(),
        "target_scale": target_scale,
        "seed": SEED,
        "best_epoch": best_epoch,
        "epochs": history,
        "seconds": perf_counter() - started,
    }
    if not daily:
        metadata.update(
            feature_means=scaler.mean_.tolist(), feature_scales=scaler.scale_.tolist()
        )
    return predicted, best_state, metadata

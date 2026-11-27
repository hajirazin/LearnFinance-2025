"""Fit raw and MAD price-input PatchTST models on the frozen matched rows."""

import argparse
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from brain_api.core.patchtst.config import DEFAULT_CONFIG
from compare_patchtst import sha256
from mad_experiment_data import ARMS, OUTPUT, load_arrays, prepare
from train_v1_xgboost_comparison import (
    make_patchtst,
    patch_predictions,
    weekly_weights,
    write_json,
)

SEED = 42
EPOCHS = 60
BATCH = 256
PATIENCE = 8


def fit(arm, manifest, device):
    specification = {
        "dataset_sha256": sha256(OUTPUT / "dataset.json"),
        "trainer_sha256": sha256(Path(__file__)),
        "shared_trainer_sha256": sha256(
            Path(__file__).parents[1]
            / "AlphaAllocationV1/train_v1_xgboost_comparison.py"
        ),
        "seed": SEED,
        "arm": arm,
        "epochs_cap": EPOCHS,
        "batch_size": BATCH,
        "patience": PATIENCE,
        "learning_rate": DEFAULT_CONFIG.learning_rate,
        "target_scale": manifest["target_std"],
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("numpy", "pandas", "torch", "transformers")
        },
    }
    metadata_path = OUTPUT / f"{arm}.json"
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text())
        if old["specification"] != specification:
            raise ValueError(f"Changed fit specification for {arm}")
        for filename, digest in old["artifacts"].items():
            if sha256(OUTPUT / filename) != digest:
                raise ValueError(f"Modified artifact {filename}")
        print(f"Reused verified {arm} fit", flush=True)
        return
    torch.manual_seed(SEED)
    model = make_patchtst().to(device)
    x, y, _ = load_arrays("train", arm)
    validation, target, index = load_arrays("validation", arm)
    scale = manifest["target_std"]
    weights = weekly_weights(index)
    optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_CONFIG.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5
    )
    rng = np.random.default_rng(SEED)
    best, best_epoch, history = np.inf, 0, []
    path = OUTPUT / f"{arm}.pt"
    for epoch in range(1, EPOCHS + 1):
        started, total = perf_counter(), 0.0
        model.train()
        order = rng.permutation(len(y))
        for start in range(0, len(y), BATCH):
            rows = order[start : start + BATCH]
            batch = torch.tensor(np.asarray(x[rows]), device=device)
            actual = torch.tensor(y[rows] / scale, device=device)
            optimizer.zero_grad(set_to_none=True)
            predicted = model(past_values=batch).regression_outputs[:, 0]
            loss = torch.nn.functional.mse_loss(predicted, actual)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss for {arm}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(rows) * scale**2
        predicted = patch_predictions(model, validation, device) * scale
        mse = float(np.average((predicted - target) ** 2, weights=weights))
        if not np.isfinite(mse):
            raise FloatingPointError(f"Nonfinite validation MSE for {arm}")
        scheduler.step(mse)
        history.append(
            {
                "epoch": epoch,
                "training_mse": total / len(y),
                "validation_mse": mse,
                "seconds": perf_counter() - started,
            }
        )
        if mse < best:
            best, best_epoch = mse, epoch
            torch.save(
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                path,
            )
        write_json(
            OUTPUT / f"{arm}_progress.json",
            {"history": history, "best_epoch": best_epoch},
        )
        print(
            f"{arm}: epoch {epoch}, validation MSE {mse:.8f}, best {best_epoch}, {perf_counter() - started:.1f}s",
            flush=True,
        )
        if epoch - best_epoch >= PATIENCE:
            break
    model.load_state_dict(torch.load(path, weights_only=True, map_location=device))
    test, _, _ = load_arrays("test", arm)
    predictions = OUTPUT / f"{arm}_predictions.npy"
    np.save(predictions, patch_predictions(model, test, device) * scale)
    write_json(
        metadata_path,
        {
            "specification": specification,
            "config": model.config.to_dict(),
            "device": str(device),
            "best_epoch": best_epoch,
            "validation_mse": best,
            "history": history,
            "artifacts": {p.name: sha256(p) for p in (path, predictions)},
        },
    )
    if device.type == "mps":
        del model
        torch.mps.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    torch.empty(0, device=device)
    manifest = prepare()
    for arm in ARMS if args.arm is None else (args.arm,):
        fit(arm, manifest, device)


if __name__ == "__main__":
    main()

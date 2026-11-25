"""Fit matched seven-feature weekly ridge, XGBoost, and PatchTST forecasters."""

import argparse
import importlib.metadata
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from brain_api.core.patchtst.config import DEFAULT_CONFIG
from compare_patchtst import sha256
from compare_ridge_lookback import PENALTIES, RidgeMoments, design
from transformers import PatchTSTForRegression
from v1_hypothesis_data import (
    DATA,
    ROSTER,
    SEVEN,
    VALIDATION,
    WIDTH,
    daily_features,
    decision_dates,
    session_data,
    windows,
)
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data/xgboost_comparison"
SEEDS = (42, 43, 44)


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def purged_dates(sessions):
    anchors = decision_dates(sessions, evaluation=False)
    ends = sessions[sessions.get_indexer(anchors) + 5]
    return anchors[ends < VALIDATION], anchors[
        (anchors >= VALIDATION) & (ends < pd.Timestamp("2026-01-01"))
    ]


def shared_inputs(x, mask):
    return design(x, mask)[1].reshape(len(x), WIDTH, len(SEVEN)).astype(np.float32)


def weekly_weights(index):
    dates = index.get_level_values("friday")
    counts = dates.value_counts()
    weights = 1.0 / counts.reindex(dates).to_numpy(dtype=float)
    return weights / weights.mean()


def prepare_split(logs, features, dates, name):
    matrices, targets, keys = [], [], []
    for start in range(0, len(dates), 10):
        index, x, mask, y = windows(
            logs, features, dates[start : start + 10], use_features=True, year=True
        )
        if name != "test":
            keep = np.isfinite(y).all(axis=1)
            index, x, mask, y = index[keep], x[keep], mask[keep], y[keep]
        matrices.append(shared_inputs(x, mask))
        targets.append(np.expm1(y.sum(axis=1, dtype=np.float64)).astype(np.float32))
        frame = index.to_frame(index=False)
        ends = logs.index[logs.index.get_indexer(frame.friday) + 5]
        frame["target_end"] = ends
        keys.append(frame)
    x = np.concatenate(matrices)
    y = np.concatenate(targets)
    index = pd.concat(keys, ignore_index=True)
    if not np.isfinite(x).all() or index.duplicated(["friday", "symbol"]).any():
        raise ValueError(f"Invalid shared inputs for {name}")
    np.save(OUTPUT / f"{name}_x.npy", x)
    np.save(OUTPUT / f"{name}_y.npy", y)
    index.to_csv(OUTPUT / f"{name}_index.csv", index=False)
    print(
        f"Prepared {name}: {len(x):,} rows, {index.symbol.nunique()} stocks", flush=True
    )
    return {
        "rows": len(x),
        "symbols": index.symbol.nunique(),
        "first": str(index.friday.min().date()),
        "last": str(index.friday.max().date()),
    }


def prepare():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = [
        ROSTER,
        DATA / "daily_ohlcv_2016_2025.csv",
        DATA / "friday/flagged_daily_returns.csv",
        DATA / "news/news_v1.duckdb",
        ROOT / "data/v1_holdout/prices_2026.csv",
        ROOT / "data/v1_holdout/news/news_v1.duckdb",
    ]
    sources = {str(path): sha256(path) for path in paths}
    manifest_path = OUTPUT / "dataset.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["sources"] != sources:
            raise ValueError("Frozen dataset sources changed")
        for path, digest in manifest["arrays"].items():
            if sha256(OUTPUT / path) != digest:
                raise ValueError(f"Cached dataset changed: {path}")
        return manifest
    calendar, _, close, volume, logs, _ = session_data()
    train, validation = purged_dates(logs.index)
    features = daily_features(calendar, close, volume, logs, evaluation=False)
    counts = {
        name: prepare_split(logs, features, dates, name)
        for name, dates in (("train", train), ("validation", validation))
    }
    del features
    features = daily_features(calendar, close, volume, logs, evaluation=True)
    counts["test"] = prepare_split(
        logs, features, decision_dates(logs.index, evaluation=True), "test"
    )
    if counts["train"]["rows"] != 82399 or counts["validation"]["rows"] != 20938:
        raise ValueError("Shared examples drifted from the ridge archive")
    manifest = {
        "sources": sources,
        "splits": counts,
        "features": list(SEVEN),
        "shape": [WIDTH, len(SEVEN)],
        "seeds": list(SEEDS),
        "target_std": float(np.std(np.load(OUTPUT / "train_y.npy"), dtype=np.float64)),
        "target_mean": float(
            np.mean(np.load(OUTPUT / "train_y.npy"), dtype=np.float64)
        ),
        "arrays": {
            path.name: sha256(path)
            for path in OUTPUT.iterdir()
            if path.suffix in (".npy", ".csv")
        },
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "command": " ".join(sys.argv),
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "pandas",
                "torch",
                "transformers",
                "xgboost",
                "duckdb",
            )
        },
        "normalization": "Per-window observed mean/std with variance epsilon 1e-5, masked zero",
        "scope": "Frozen historical roster and previously inspected evaluation period",
    }
    write_json(manifest_path, manifest)
    return manifest


def load_split(name):
    x = np.load(OUTPUT / f"{name}_x.npy", mmap_mode="r")
    y = np.load(OUTPUT / f"{name}_y.npy")
    index = pd.read_csv(
        OUTPUT / f"{name}_index.csv", parse_dates=["friday", "target_end"]
    )
    return x, y, pd.MultiIndex.from_frame(index[["friday", "symbol"]])


def completed(name, *, output=OUTPUT):
    path = output / f"{name}.json"
    if not path.exists():
        return False
    metadata = json.loads(path.read_text())
    for filename, digest in metadata["artifacts"].items():
        if sha256(output / filename) != digest:
            raise ValueError(f"Incomplete or modified fit: {name}/{filename}")
    if metadata["dataset_sha256"] != sha256(OUTPUT / "dataset.json"):
        raise ValueError(f"Dataset mismatch for {name}")
    return True


def finish(name, metadata, artifacts, *, output=OUTPUT):
    write_json(
        output / f"{name}.json",
        {
            **metadata,
            "dataset_sha256": sha256(OUTPUT / "dataset.json"),
            "runner_sha256": sha256(Path(__file__)),
            "artifacts": {path.name: sha256(path) for path in artifacts},
        },
    )


def fit_ridge():
    if completed("ridge"):
        return
    x, y, _ = load_split("train")
    v, target, index = load_split("validation")
    moments = RidgeMoments(WIDTH * len(SEVEN))
    for start in range(0, len(y), 4096):
        moments.update(
            x[start : start + 4096].reshape(-1, WIDTH * len(SEVEN)),
            y[start : start + 4096],
        )
    weights, bias = moments.coefficients(PENALTIES)
    errors = v.reshape(len(v), -1) @ weights + bias - target[:, None]
    mse = np.average(errors**2, axis=0, weights=weekly_weights(index))
    best = int(mse.argmin())
    path = OUTPUT / "ridge.npz"
    np.savez(path, weights=weights[:, best], bias=bias[best])
    test, _, _ = load_split("test")
    forecast = OUTPUT / "ridge_predictions.npy"
    np.save(forecast, test.reshape(len(test), -1) @ weights[:, best] + bias[best])
    finish(
        "ridge",
        {
            "penalty": float(PENALTIES[best]),
            "validation_mse": float(mse[best]),
            "grid": dict(zip(map(str, PENALTIES), map(float, mse), strict=True)),
        },
        [path, forecast],
    )
    print(
        f"Ridge selected penalty {PENALTIES[best]}, validation MSE {mse[best]:.8f}",
        flush=True,
    )


def fit_xgboost(seed, scale):
    name = f"xgboost_{seed}"
    if completed(name):
        return
    x, y, _ = load_split("train")
    v, target, index = load_split("validation")
    depths = (
        (2, 4)
        if seed == 42
        else (json.loads((OUTPUT / "xgboost_42.json").read_text())["depth"],)
    )
    candidates = []
    for depth in depths:
        started = perf_counter()
        model = XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            max_depth=depth,
            learning_rate=0.03,
            n_estimators=600,
            min_child_weight=20,
            reg_lambda=10,
            subsample=0.8,
            colsample_bytree=0.8,
            max_bin=128,
            n_jobs=4,
            random_state=seed,
            eval_metric="rmse",
            early_stopping_rounds=30,
        )
        model.fit(
            x.reshape(len(x), -1),
            y / scale,
            eval_set=[(v.reshape(len(v), -1), target / scale)],
            sample_weight_eval_set=[weekly_weights(index)],
            verbose=False,
        )
        prediction = model.predict(v.reshape(len(v), -1)) * scale
        mse = float(
            np.average((prediction - target) ** 2, weights=weekly_weights(index))
        )
        candidates.append((mse, depth, model))
        print(
            f"XGBoost seed {seed}, depth {depth}, best tree {model.best_iteration + 1}, "
            f"validation MSE {mse:.8f}, {perf_counter() - started:.1f}s",
            flush=True,
        )
    mse, depth, model = min(candidates, key=lambda candidate: candidate[:2])
    path = OUTPUT / f"{name}.ubj"
    model.get_booster().save_model(path)
    test, _, _ = load_split("test")
    forecast = OUTPUT / f"{name}_predictions.npy"
    np.save(forecast, model.predict(test.reshape(len(test), -1)) * scale)
    finish(
        name,
        {
            "depth": depth,
            "seed": seed,
            "target_scale": scale,
            "validation_mse": mse,
            "best_iteration": model.best_iteration,
            "depth_scores": {str(d): score for score, d, _ in candidates},
            "history": model.evals_result(),
            "booster_config": json.loads(model.get_booster().save_config()),
        },
        [path, forecast],
    )


def make_patchtst():
    config = replace(
        DEFAULT_CONFIG, context_length=WIDTH, num_input_channels=7, prediction_length=1
    ).to_hf_config()
    config.scaling = None
    config.num_targets = 1
    return PatchTSTForRegression(config)


def patch_input(values, device, *, disable_news=False):
    """Copy the batch and optionally remove sentiment information only."""
    batch = torch.tensor(np.asarray(values), device=device)
    if disable_news:
        batch[..., list(SEVEN).index("news_sentiment")] = 0
    return batch


def patch_predictions(model, x, device, *, disable_news=False):
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(x), 256):
            batch = patch_input(
                x[start : start + 256], device, disable_news=disable_news
            )
            predictions.append(
                model(past_values=batch).regression_outputs[:, 0].cpu().numpy()
            )
    return np.concatenate(predictions)


def fit_patchtst(seed, scale, device, *, output=OUTPUT, disable_news=False):
    name = f"patchtst_{seed}"
    if completed(name, output=output):
        metadata = json.loads((output / f"{name}.json").read_text())
        if metadata.get("disable_news", False) != disable_news:
            raise ValueError(f"Sentiment policy mismatch for {output / name}")
        return
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    model = make_patchtst().to(device)
    x, y, _ = load_split("train")
    v, target, index = load_split("validation")
    validation_weights = weekly_weights(index)
    optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_CONFIG.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5
    )
    rng = np.random.default_rng(seed)
    best, best_epoch, history = np.inf, 0, []
    path = output / f"{name}.pt"
    for epoch in range(1, 61):
        started = perf_counter()
        model.train()
        total = 0.0
        order = rng.permutation(len(y))
        for start in range(0, len(y), 256):
            rows = order[start : start + 256]
            batch = patch_input(x[rows], device, disable_news=disable_news)
            actual = torch.tensor(y[rows] / scale, device=device)
            optimizer.zero_grad(set_to_none=True)
            predicted = model(past_values=batch).regression_outputs[:, 0]
            loss = torch.nn.functional.mse_loss(predicted, actual)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss for {name}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(rows) * scale**2
        predicted = (
            patch_predictions(model, v, device, disable_news=disable_news) * scale
        )
        mse = float(np.average((predicted - target) ** 2, weights=validation_weights))
        if not np.isfinite(mse):
            raise FloatingPointError(f"Nonfinite validation for {name}")
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
                {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
                path,
            )
        write_json(
            output / f"{name}_progress.json",
            {"history": history, "best_epoch": best_epoch},
        )
        print(
            f"PatchTST seed {seed}, epoch {epoch}, validation MSE {mse:.8f}, "
            f"best epoch {best_epoch}, {perf_counter() - started:.1f}s",
            flush=True,
        )
        if epoch - best_epoch >= 8:
            break
    model.load_state_dict(torch.load(path, weights_only=True, map_location=device))
    test, _, _ = load_split("test")
    forecast = output / f"{name}_predictions.npy"
    np.save(
        forecast,
        patch_predictions(model, test, device, disable_news=disable_news) * scale,
    )
    finish(
        name,
        {
            "seed": seed,
            "disable_news": disable_news,
            "target_scale": scale,
            "device": str(device),
            "best_epoch": best_epoch,
            "validation_mse": best,
            "history": history,
            "config": model.config.to_dict(),
        },
        [path, forecast],
        output=output,
    )
    del model
    if device.type == "mps":
        torch.mps.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("prepare", "ridge", "xgboost", "patchtst", "all"),
        default="all",
    )
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    manifest = prepare()
    scale = manifest["target_std"]
    if args.stage in ("ridge", "all"):
        fit_ridge()
    if args.stage in ("xgboost", "all"):
        for seed in SEEDS if args.seed is None else (args.seed,):
            fit_xgboost(seed, scale)
    if args.stage in ("patchtst", "all"):
        for seed in SEEDS if args.seed is None else (args.seed,):
            fit_patchtst(seed, scale, torch.device(args.device))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the frozen full-universe PatchTST control/candidate comparison."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import torch
from experiment_data import (
    fetch_uncached_halal_new_universe,
    load_halal_new_universe_cache,
    load_or_download_prices,
)
from experiment_metrics import (
    aggregate_metrics,
    paired_block_bootstrap,
    weekly_metrics,
)
from experiment_panel import SplitWindow, build_weekly_panel, panel_arrays
from experiment_spec import (
    ARMS,
    CONTEXT_LENGTH,
    PREDICTION_LENGTH,
    SEEDS,
    TOP_K,
    arm_manifest,
    json_dump,
    runtime_manifest,
    sha256_file,
    sha256_json,
)
from experiment_training import (
    predict_log_returns,
    prediction_frame,
    train_arm,
)
from sklearn.linear_model import Ridge

BASE = Path(__file__).resolve().parent
REPOSITORY = BASE.parents[1]
DATA_DIR = BASE / "data"
MODEL_DIR = BASE / "models"
RESULTS_DIR = BASE / "results"
UNIVERSE_PATH = DATA_DIR / "halal_new_universe.json"

SPLITS = {
    "train": SplitWindow(date(2015, 5, 4), date(2022, 12, 19)),
    "validation": SplitWindow(date(2023, 1, 9), date(2023, 12, 18)),
    "test": SplitWindow(date(2024, 1, 8), date(2025, 12, 22)),
}
DATA_START = date(2015, 1, 1)
DATA_END_EXCLUSIVE = date(2026, 1, 6)


def _git_value(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPOSITORY, text=True
    ).strip()


def _load_or_fetch_universe(
    universe_cache: Path | None,
) -> tuple[list[str], dict[str, Any]]:
    if UNIVERSE_PATH.exists():
        manifest = json.loads(UNIVERSE_PATH.read_text())
        symbols = [row["symbol"] for row in manifest["stocks"]]
        if len(symbols) != manifest["halal_new_count"]:
            raise RuntimeError("universe manifest count mismatch")
        return symbols, manifest
    if universe_cache is not None:
        symbols, manifest = load_halal_new_universe_cache(universe_cache)
        json_dump(UNIVERSE_PATH, manifest)
        return symbols, manifest
    symbols, manifest = fetch_uncached_halal_new_universe(
        brain_env_path=REPOSITORY / "brain_api" / ".env"
    )
    json_dump(UNIVERSE_PATH, manifest)
    return symbols, manifest


def _panel_identity(panel: pd.DataFrame, include_test: bool) -> str:
    rows = panel[panel["split"] != "test"] if not include_test else panel
    identity = rows[["decision_date", "split", "symbol", "context_end", "target_end"]]
    return sha256_json(identity.astype(str).to_dict(orient="records"))


def _ensemble(frames: list[pd.DataFrame], arm: str) -> pd.DataFrame:
    stacked = pd.concat(frames, ignore_index=True)
    result = stacked.groupby(["decision_date", "symbol"], as_index=False).agg(
        actual_weekly_return=("actual_weekly_return", "first"),
        predicted_weekly_return=("predicted_weekly_return", "mean"),
    )
    result["arm"] = arm
    result["seed"] = "ensemble"
    return result


def _control_frames(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    fit = panel[panel["split"].isin(["train", "validation"])].copy()
    test = panel[panel["split"] == "test"].copy()
    if (
        fit["actual_weekly_log_return"].isna().any()
        or test["actual_weekly_log_return"].isna().any()
    ):
        raise RuntimeError("control labels are locked")
    feature_columns = [
        "past_week_log_return",
        "momentum_4w_log_return",
        "context_log_return",
        "volatility_4w",
        "volume_volatility_4w",
    ]
    ridge = Ridge(alpha=1.0)
    ridge.fit(fit[feature_columns], fit["actual_weekly_log_return"])
    prediction_logs: dict[str, np.ndarray] = {
        "zero_return": np.zeros(len(test)),
        "persistence_1w": test["past_week_log_return"].to_numpy(float),
        "reversal_1w": -test["past_week_log_return"].to_numpy(float),
        "momentum_4w": test["momentum_4w_log_return"].to_numpy(float),
        "ridge": ridge.predict(test[feature_columns]),
    }
    causal_mean = np.empty(len(test), dtype=float)
    majority_sign = np.empty(len(test), dtype=float)
    symbol_history = {
        symbol: values["actual_weekly_log_return"].to_list()
        for symbol, values in fit.groupby("symbol")
    }
    global_history = fit["actual_weekly_log_return"].to_list()
    for _, indices in test.groupby("decision_date", sort=True).groups.items():
        magnitude = float(np.median(np.abs(global_history)))
        sign = 1.0 if np.mean(np.asarray(global_history) > 0) >= 0.5 else -1.0
        for index in indices:
            history = symbol_history.get(test.at[index, "symbol"], [])
            causal_mean[test.index.get_loc(index)] = (
                float(np.mean(history)) if history else 0.0
            )
            majority_sign[test.index.get_loc(index)] = sign * magnitude
        for index in indices:
            value = float(test.at[index, "actual_weekly_log_return"])
            symbol_history.setdefault(test.at[index, "symbol"], []).append(value)
            global_history.append(value)
    prediction_logs["causal_historical_mean"] = causal_mean
    prediction_logs["causal_majority_sign"] = majority_sign

    metadata = test[
        ["decision_date", "symbol", "actual_weekly_log_return"]
    ].reset_index(drop=True)
    frames: dict[str, pd.DataFrame] = {}
    for name, predictions in prediction_logs.items():
        frame = prediction_frame(metadata, np.asarray(predictions))
        frame["arm"] = name
        frame["seed"] = "control"
        frames[name] = frame
    return frames


def _source_hashes() -> dict[str, str]:
    return {
        path.name: sha256_file(path)
        for path in BASE.glob("*.py")
        if path.name != "test_experiment.py"
    }


def run(
    *,
    max_epochs: int,
    patience: int,
    device_name: str,
    universe_cache: Path | None,
) -> None:
    started = time.perf_counter()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    attempt_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    device = torch.device(device_name)
    json_dump(
        RESULTS_DIR / "status.json",
        {"status": "in_progress", "attempt_id": attempt_id},
    )

    symbols, universe_manifest = _load_or_fetch_universe(universe_cache)
    prices, price_manifest = load_or_download_prices(
        symbols,
        data_dir=DATA_DIR,
        start_date=DATA_START,
        end_date=DATA_END_EXCLUSIVE,
    )
    calendar = xcals.get_calendar("XNYS")
    sessions = calendar.sessions_in_range(DATA_START, DATA_END_EXCLUSIVE)
    locked_panel = build_weekly_panel(
        prices,
        sessions=sessions,
        splits=SPLITS,
        include_test_labels=False,
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
    )
    train_x, train_y, _train_metadata = panel_arrays(locked_panel, "train")
    validation_x, validation_y, validation_metadata = panel_arrays(
        locked_panel, "validation"
    )
    locked_non_test_identity = _panel_identity(locked_panel, include_test=False)
    models: dict[tuple[str, int], torch.nn.Module] = {}
    training_metadata: dict[str, Any] = {}
    validation_frames: list[pd.DataFrame] = []

    for arm_name, arm in ARMS.items():
        for seed in SEEDS:
            print(f"TRAIN arm={arm_name} seed={seed} device={device}", flush=True)
            model, metadata = train_arm(
                arm,
                seed,
                train_x,
                train_y,
                validation_x,
                validation_y,
                validation_metadata,
                model_dir=MODEL_DIR / arm_name / str(seed),
                device=device,
                max_epochs=max_epochs,
                patience=patience,
            )
            models[(arm_name, seed)] = model
            training_metadata[f"{arm_name}/{seed}"] = metadata
            predictions = predict_log_returns(
                model.to(device), validation_x, device=device
            )
            model.cpu()
            frame = prediction_frame(validation_metadata, predictions)
            frame["arm"] = arm_name
            frame["seed"] = str(seed)
            validation_frames.append(frame)

    validation_results = {
        key: aggregate_metrics(
            pd.concat(
                [
                    frame
                    for frame in validation_frames
                    if f"{frame['arm'].iloc[0]}/{frame['seed'].iloc[0]}" == key
                ]
            ),
            top_k=TOP_K,
        )
        for key in training_metadata
    }
    json_dump(
        RESULTS_DIR / "test_unlock.json",
        {
            "attempt_id": attempt_id,
            "unlocked_at_utc": datetime.now(UTC).isoformat(),
            "test_labels_read_before_unlock": False,
            "non_test_panel_sha256": locked_non_test_identity,
            "validation_results": validation_results,
        },
    )

    panel = build_weekly_panel(
        prices,
        sessions=sessions,
        splits=SPLITS,
        include_test_labels=True,
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
    )
    if (
        _panel_identity(panel[panel["split"] != "test"], include_test=True)
        != locked_non_test_identity
    ):
        raise RuntimeError("non-test panel changed when test labels were unlocked")
    test_x, _test_y, test_metadata = panel_arrays(panel, "test")
    prediction_frames: list[pd.DataFrame] = []
    ensembles: dict[str, pd.DataFrame] = {}
    for arm_name in ARMS:
        seed_frames: list[pd.DataFrame] = []
        for seed in SEEDS:
            model = models[(arm_name, seed)].to(device)
            predictions = predict_log_returns(model, test_x, device=device)
            model.cpu()
            frame = prediction_frame(test_metadata, predictions)
            frame["arm"] = arm_name
            frame["seed"] = str(seed)
            seed_frames.append(frame)
            prediction_frames.append(frame)
        ensembles[arm_name] = _ensemble(seed_frames, arm_name)
        prediction_frames.append(ensembles[arm_name])
    controls = _control_frames(panel)
    prediction_frames.extend(controls.values())

    metrics: dict[str, Any] = {}
    for frame in prediction_frames:
        key = f"{frame['arm'].iloc[0]}/{frame['seed'].iloc[0]}"
        metrics[key] = aggregate_metrics(frame, top_k=TOP_K)
    candidate = ensembles["coherent_candidate_10_5"]
    control = ensembles["corrected_control_16_8"]
    paired = {
        "candidate_minus_corrected_control": paired_block_bootstrap(
            candidate,
            control,
            seed=20260823,
            repetitions=2000,
            block_weeks=4,
            top_k=TOP_K,
        )
    }
    for name, frame in controls.items():
        paired[f"candidate_minus_{name}"] = paired_block_bootstrap(
            candidate,
            frame,
            seed=20260823,
            repetitions=2000,
            block_weeks=4,
            top_k=TOP_K,
        )

    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    all_predictions.to_csv(RESULTS_DIR / "predictions.csv", index=False)
    weekly = []
    for frame in prediction_frames:
        result = weekly_metrics(frame, top_k=TOP_K)
        result["arm"] = frame["arm"].iloc[0]
        result["seed"] = frame["seed"].iloc[0]
        weekly.append(result)
    pd.concat(weekly, ignore_index=True).to_csv(
        RESULTS_DIR / "weekly_metrics.csv", index=False
    )

    panel_counts = {
        split: {
            "rows": len(values),
            "weeks": int(values["decision_date"].nunique()),
            "unique_symbols": int(values["symbol"].nunique()),
            "min_symbols_per_week": int(
                values.groupby("decision_date")["symbol"].count().min()
            ),
            "median_symbols_per_week": float(
                values.groupby("decision_date")["symbol"].count().median()
            ),
            "max_symbols_per_week": int(
                values.groupby("decision_date")["symbol"].count().max()
            ),
            "first_decision": str(values["decision_date"].min()),
            "last_decision": str(values["decision_date"].max()),
        }
        for split, values in panel.groupby("split")
    }
    manifest = {
        "status": "complete",
        "attempt_id": attempt_id,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "git_head": _git_value("rev-parse", "HEAD"),
        "git_status_short": _git_value("status", "--short"),
        "production_current_pointers_touched": False,
        "runtime_seconds": time.perf_counter() - started,
        "runtime": runtime_manifest(),
        "device_used": str(device),
        "seeds": SEEDS,
        "splits": {
            name: {"start": str(window.start), "end": str(window.end)}
            for name, window in SPLITS.items()
        },
        "data_window": {
            "start": str(DATA_START),
            "end_exclusive": str(DATA_END_EXCLUSIVE),
        },
        "universe": universe_manifest,
        "price_data": price_manifest,
        "requested_symbol_count": len(symbols),
        "downloaded_symbol_count": len(prices),
        "panel_counts": panel_counts,
        "panel_exclusion_counts": panel.attrs["exclusion_counts"],
        "per_symbol_panel_counts": panel.attrs["per_symbol_counts"],
        "arms": arm_manifest(),
        "training": training_metadata,
        "metrics": metrics,
        "paired_block_bootstrap": paired,
        "bootstrap": {"repetitions": 2000, "block_weeks": 4, "unit": "decision_week"},
        "source_sha256": _source_hashes(),
        "artifact_sha256": {
            path.name: sha256_file(path)
            for path in [
                RESULTS_DIR / "predictions.csv",
                RESULTS_DIR / "weekly_metrics.csv",
                RESULTS_DIR / "test_unlock.json",
            ]
        },
        "limitations": [
            "Current 2026 halal_new membership is used historically; results have survivorship and universe-membership bias.",
            "Yahoo adjusted OHLCV is revision-prone and is not institutional point-in-time data.",
            "This is one static chronological holdout, not expanding-window production retraining.",
            "Top-15 spreads are gross signal diagnostics; they omit sticky selection, HRP weights, costs, and execution.",
            "Only two predeclared model configurations and three seeds are compared.",
        ],
    }
    json_dump(RESULTS_DIR / "manifest.json", manifest)
    json_dump(
        RESULTS_DIR / "status.json", {"status": "complete", "attempt_id": attempt_id}
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--universe-cache", type=Path)
    arguments = parser.parse_args()
    run(
        max_epochs=arguments.max_epochs,
        patience=arguments.patience,
        device_name=arguments.device,
        universe_cache=arguments.universe_cache,
    )


if __name__ == "__main__":
    main()

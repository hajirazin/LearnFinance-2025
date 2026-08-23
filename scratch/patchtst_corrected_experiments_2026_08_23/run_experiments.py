#!/usr/bin/env python3
"""Run the three approved, pre-registered PatchTST experiments end to end."""

from __future__ import annotations

import contextlib
import fcntl
import json
import math
import os
import subprocess
import sys
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.linear_model import Ridge

from brain_api.core.prices import load_prices_yfinance
from evaluation import (
    aggregate_metrics,
    bootstrap_intervals,
    paired_bootstrap_delta,
    weekly_statistics,
)
from experiment_core import (
    ARCHITECTURES,
    SEEDS,
    SPLITS,
    SYMBOLS,
    architecture_manifest,
    build_weekly_panel,
    control_predictions,
    json_dump,
    load_model_artifact,
    model_sensitivity,
    panel_arrays,
    predict,
    ridge_predictions,
    sha256_file,
    train_model,
    training_fingerprint,
)


BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
MODELS = BASE / "models"
RESULTS = BASE / "results"
LOG = BASE / "run.log"
BASELINE_COMMIT = "f3c4dbefe822b50efc80846dad00aa072b51c68c"


class Tee:
    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=BASE.parents[1], text=True
    ).strip()


def _require_clean_tracked_baseline() -> None:
    repository = BASE.parents[1]
    if _git_head() != BASELINE_COMMIT:
        raise RuntimeError(
            f"HEAD differs from preregistered baseline {BASELINE_COMMIT}"
        )
    for args in (
        ["git", "diff", "--quiet", BASELINE_COMMIT, "--"],
        ["git", "diff", "--cached", "--quiet", BASELINE_COMMIT, "--"],
    ):
        if subprocess.run(args, cwd=repository, check=False).returncode != 0:
            raise RuntimeError(
                "tracked worktree/index differs from the preregistered baseline"
            )


def _load_or_download_prices() -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    DATA.mkdir(parents=True, exist_ok=True)
    cache_manifest = DATA / "adjusted_ohlcv_manifest.json"
    cached: dict[str, pd.DataFrame] = {}
    if cache_manifest.exists():
        meta = json.loads(cache_manifest.read_text())
        if meta.get("auto_adjust") is not True:
            raise RuntimeError("cached data is not explicitly adjusted")
        if meta.get("requested_symbols") != SYMBOLS or not set(
            meta.get("files", {})
        ).issubset(SYMBOLS):
            raise RuntimeError(
                "adjusted cache does not match the frozen diagnostic universe"
            )
        referenced_files = [info.get("file") for info in meta.get("files", {}).values()]
        if len(referenced_files) != len(set(referenced_files)):
            raise RuntimeError(
                "adjusted cache manifest aliases multiple symbols to one file"
            )
        for symbol, info in meta["files"].items():
            path = DATA / info["file"]
            if not path.exists() or sha256_file(path) != info["sha256"]:
                raise RuntimeError(f"adjusted cache hash mismatch: {symbol}")
            frame = pd.read_csv(path, index_col=0, parse_dates=True)
            cached[symbol] = frame
        if len(cached) >= 10:
            print(f"Loaded verified adjusted cache for {len(cached)} symbols")
            return cached, meta

    prices = load_prices_yfinance(
        SYMBOLS,
        date(2015, 1, 1),
        date(2026, 1, 5),
        log_prefix="[CorrectedPatchTST]",
    )
    if len(prices) < 10:
        raise RuntimeError(
            f"adjusted download left fewer than 10 symbols: {sorted(prices)}"
        )
    files: dict[str, Any] = {}
    normalized: dict[str, pd.DataFrame] = {}
    for symbol, frame in sorted(prices.items()):
        clean = frame[["open", "high", "low", "close", "volume"]].copy()
        clean.index = pd.to_datetime(clean.index).tz_localize(None)
        clean = clean.sort_index().loc[:"2025-12-31"].dropna()
        path = DATA / f"{symbol}_adjusted.csv"
        clean.to_csv(path, index_label="date", float_format="%.12g")
        normalized[symbol] = pd.read_csv(path, index_col=0, parse_dates=True)
        files[symbol] = {
            "file": path.name,
            "sha256": sha256_file(path),
            "rows": len(clean),
            "start": str(clean.index.min().date()),
            "end": str(clean.index.max().date()),
        }
    meta = {
        "provider": "yfinance",
        "auto_adjust": True,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_symbols": SYMBOLS,
        "files": files,
    }
    json_dump(cache_manifest, meta)
    return normalized, meta


def _prediction_frame(
    panel: pd.DataFrame, split: str, values: np.ndarray, arm: str, seed: str | int
) -> pd.DataFrame:
    rows = panel[panel["split"] == split][
        ["decision_date", "symbol", "actual_weekly_return"]
    ].copy()
    if len(rows) != len(values):
        raise RuntimeError(f"prediction length mismatch for {arm}/{seed}")
    # The models and controls operate on summed log returns. Convert at the
    # reporting boundary so magnitude/economic metrics match production's
    # compounded simple-return contract; ranks and signs are unchanged.
    rows["actual_weekly_return"] = np.expm1(
        rows["actual_weekly_return"].to_numpy(float)
    )
    rows["predicted_weekly_return"] = np.expm1(np.asarray(values, dtype=float))
    rows["arm"] = arm
    rows["seed"] = str(seed)
    return rows


def _ridge_for_validation(panel: pd.DataFrame) -> np.ndarray:
    train = panel[panel["split"] == "train"]
    validation = panel[panel["split"] == "validation"]
    model = Ridge(alpha=1.0)
    model.fit(
        np.stack(train["ridge_features"]), train["actual_weekly_return"].to_numpy()
    )
    return model.predict(np.stack(validation["ridge_features"]))


def _ensemble(frames: list[pd.DataFrame], arm: str, split: str) -> pd.DataFrame:
    stacked = pd.concat(frames, ignore_index=True)
    result = stacked.groupby(["decision_date", "symbol"], as_index=False).agg(
        actual_weekly_return=("actual_weekly_return", "first"),
        predicted_weekly_return=("predicted_weekly_return", "mean"),
    )
    result["arm"] = arm
    result["seed"] = "aggregate"
    result["split"] = split
    return result


def _select_with_declared_tie_break(scores: dict[str, float], order: list[str]) -> str:
    if set(scores) != set(order):
        raise ValueError(
            "selection candidates differ from the declared tie-break order"
        )
    best = max(scores.values())
    return next(
        name
        for name in order
        if math.isclose(scores[name], best, rel_tol=0.0, abs_tol=1e-12)
    )


def _record_failure(
    failure: dict[str, str],
    *,
    attempt_id: str,
    current: list[dict[str, str]],
    history: list[dict[str, str]],
) -> None:
    current.append(failure)
    history.append(
        {
            "attempt_id": attempt_id,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            **failure,
        }
    )
    json_dump(RESULTS / "failures.json", current)
    json_dump(RESULTS / "failure_history.json", history)


def _train_or_resume(
    architecture: Any,
    seed: int,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    objective: str,
    model_dir: Path,
    group_size: int,
) -> tuple[Any, dict[str, Any]]:
    fingerprint = training_fingerprint(
        architecture, objective, (train_x, train_y, val_x, val_y)
    )
    if (model_dir / "weights.pt").exists() and (model_dir / "meta.json").exists():
        try:
            model, meta = load_model_artifact(architecture, model_dir)
            if (
                meta.get("seed") == seed
                and meta.get("objective") == objective
                and meta.get("training_fingerprint") == fingerprint
            ):
                print(f"RESUME {objective} {architecture.name} seed={seed}")
                return model, meta
            print(f"RETRAIN stale artifact {objective} {architecture.name} seed={seed}")
        except (RuntimeError, ValueError) as exc:
            print(
                f"RETRAIN invalid artifact {objective} {architecture.name} seed={seed}: {exc}"
            )
    return train_model(
        architecture,
        seed,
        train_x,
        train_y,
        val_x,
        val_y,
        objective=objective,
        group_size=group_size,
        model_dir=model_dir,
        fingerprint=fingerprint,
    )


def run() -> int:
    for directory in [DATA, MODELS, RESULTS]:
        directory.mkdir(parents=True, exist_ok=True)
    attempt_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    failures: list[dict[str, str]] = []
    failure_history_path = RESULTS / "failure_history.json"
    failure_history = (
        json.loads(failure_history_path.read_text())
        if failure_history_path.exists()
        else []
    )
    print("=== corrected PatchTST experiment suite ===")
    print(f"attempt_id={attempt_id}")
    _require_clean_tracked_baseline()
    json_dump(
        RESULTS / "summary.json",
        {
            "status": "in_progress",
            "attempt_id": attempt_id,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    prices, provenance = _load_or_download_prices()

    config = {
        "baseline_commit": BASELINE_COMMIT,
        "attempt_id": attempt_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbols_requested": SYMBOLS,
        "symbols_available": sorted(prices),
        "seeds": SEEDS,
        "splits": {
            name: [str(start), str(end)] for name, (start, end) in SPLITS.items()
        },
        "architecture": architecture_manifest(),
        "training": {
            "context": 60,
            "horizon": 5,
            "patch": 16,
            "d_model": 64,
            "heads": 4,
            "layers": 2,
            "ffn": 128,
            "learning_rate": 3e-4,
            "weight_decay": 0.0,
            "max_epochs": 60,
            "patience": 8,
        },
        "selection_policy": "highest mean validation weekly rank IC across all three declared seeds; no test access",
        "selection_tie_break": "predeclared candidate order, used only for exactly equal mean validation rank IC",
        "reported_return_unit": "compounded simple return (expm1 of summed log return)",
        "bootstrap": {
            "unit": "decision_week",
            "moving_block_weeks": 4,
            "repetitions": 2000,
        },
        "data": provenance,
        "dependencies": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    json_dump(BASE / "config.json", config)

    locked = build_weekly_panel(prices, SPLITS, include_test_labels=False)
    row_counts = locked.groupby("split").size().to_dict()
    test_locked = locked[locked["split"] == "test"]
    if (
        not test_locked["actual_weekly_return"].isna().all()
        or not test_locked["y"].isna().all()
    ):
        raise RuntimeError("test labels were materialized before unlock")
    manifest = {
        "config_sha256": sha256_file(BASE / "config.json"),
        "attempt_id": attempt_id,
        "data_manifest_sha256": sha256_file(DATA / "adjusted_ohlcv_manifest.json"),
        "row_counts": {key: int(value) for key, value in row_counts.items()},
        "week_counts": {
            name: int(locked[locked["split"] == name]["decision_date"].nunique())
            for name in SPLITS
        },
        "common_symbols": sorted(locked["symbol"].unique()),
        "exclusions": locked.attrs.get("exclusions", {}),
        "test_labels_locked": True,
        "source_sha256": {
            path.name: sha256_file(path)
            for path in [
                BASE / "run_experiments.py",
                BASE / "experiment_core.py",
                BASE / "evaluation.py",
                BASE / "test_experiments.py",
            ]
        },
    }
    if any(row_counts.get(name, 0) == 0 for name in SPLITS):
        raise RuntimeError(f"empty preregistered split: {row_counts}")
    json_dump(RESULTS / "manifest.json", manifest)
    print(f"Frozen config/manifest; locked test rows={len(test_locked)}")

    trained: dict[tuple[str, int, str], Any] = {}
    validation_frames: dict[str, list[pd.DataFrame]] = {}
    validation_seed_metrics: dict[str, dict[str, Any]] = {}
    for name, architecture in ARCHITECTURES.items():
        train_x, train_y, _ = panel_arrays(locked, "train", architecture)
        val_x, val_y, _ = panel_arrays(locked, "validation", architecture)
        validation_frames[name] = []
        validation_seed_metrics[name] = {}
        for seed in SEEDS:
            print(f"TRAIN daily_mse {name} seed={seed}")
            try:
                model, meta = _train_or_resume(
                    architecture,
                    seed,
                    train_x,
                    train_y,
                    val_x,
                    val_y,
                    objective="daily_mse",
                    model_dir=MODELS / name / str(seed),
                    group_size=len(prices),
                )
                trained[(name, seed, "daily_mse")] = model
                frame = _prediction_frame(
                    locked, "validation", predict(model, val_x), name, seed
                )
                validation_frames[name].append(frame)
                validation_seed_metrics[name][str(seed)] = {
                    "training": meta,
                    "metrics": aggregate_metrics(frame),
                }
            except Exception as exc:
                _record_failure(
                    {
                        "arm": name,
                        "seed": str(seed),
                        "stage": "daily_mse",
                        "error": repr(exc),
                    },
                    attempt_id=attempt_id,
                    current=failures,
                    history=failure_history,
                )
                raise

    channel_candidates = [
        "canonical_close_only",
        "canonical_independent_5ch",
        "canonical_mixing_5ch",
    ]
    validation_ensembles = {
        name: _ensemble(validation_frames[name], name, "validation")
        for name in ARCHITECTURES
    }
    architecture_validation_rank_ic = {
        name: float(
            np.mean(
                [
                    validation_seed_metrics[name][str(seed)]["metrics"][
                        "weekly_rank_ic"
                    ]
                    for seed in SEEDS
                ]
            )
        )
        for name in channel_candidates
    }
    if not all(np.isfinite(list(architecture_validation_rank_ic.values()))):
        raise RuntimeError(
            f"non-finite architecture selection score: {architecture_validation_rank_ic}"
        )
    selected_architecture = _select_with_declared_tie_break(
        architecture_validation_rank_ic, channel_candidates
    )
    print(f"Validation-selected architecture: {selected_architecture}")

    selected = ARCHITECTURES[selected_architecture]
    train_x, train_y, _ = panel_arrays(locked, "train", selected)
    val_x, val_y, _ = panel_arrays(locked, "validation", selected)
    listnet_name = f"{selected_architecture}__listnet"
    validation_frames[listnet_name] = []
    validation_seed_metrics[listnet_name] = {}
    for seed in SEEDS:
        print(f"TRAIN listnet {selected_architecture} seed={seed}")
        try:
            model, meta = _train_or_resume(
                selected,
                seed,
                train_x,
                train_y,
                val_x,
                val_y,
                objective="listnet",
                model_dir=MODELS / listnet_name / str(seed),
                group_size=len(prices),
            )
            trained[(selected_architecture, seed, "listnet")] = model
            frame = _prediction_frame(
                locked, "validation", predict(model, val_x), listnet_name, seed
            )
            validation_frames[listnet_name].append(frame)
            validation_seed_metrics[listnet_name][str(seed)] = {
                "training": meta,
                "metrics": aggregate_metrics(frame),
            }
        except Exception as exc:
            _record_failure(
                {
                    "arm": listnet_name,
                    "seed": str(seed),
                    "stage": "listnet",
                    "error": repr(exc),
                },
                attempt_id=attempt_id,
                current=failures,
                history=failure_history,
            )
            raise
    validation_ensembles[listnet_name] = _ensemble(
        validation_frames[listnet_name], listnet_name, "validation"
    )
    objective_candidates = [selected_architecture, listnet_name]
    objective_validation_rank_ic = {
        name: float(
            np.mean(
                [
                    validation_seed_metrics[name][str(seed)]["metrics"][
                        "weekly_rank_ic"
                    ]
                    for seed in SEEDS
                ]
            )
        )
        for name in objective_candidates
    }
    if not all(np.isfinite(list(objective_validation_rank_ic.values()))):
        raise RuntimeError(
            f"non-finite objective selection score: {objective_validation_rank_ic}"
        )
    selected_objective_arm = _select_with_declared_tie_break(
        objective_validation_rank_ic, objective_candidates
    )

    control_panel_locked = control_predictions(locked)
    validation_controls: dict[str, pd.DataFrame] = {}
    for name in [
        "zero_return",
        "historical_mean",
        "majority_sign",
        "persistence_1w",
        "reversal_1w",
        "momentum_4w",
    ]:
        validation_controls[name] = _prediction_frame(
            control_panel_locked,
            "validation",
            control_panel_locked.loc[
                control_panel_locked["split"] == "validation", name
            ].to_numpy(),
            name,
            "control",
        )
    validation_controls["ridge"] = _prediction_frame(
        locked, "validation", _ridge_for_validation(locked), "ridge", "control"
    )
    finite_control_scores = {
        name: float(aggregate_metrics(frame)["weekly_rank_ic"])
        for name, frame in validation_controls.items()
        if np.isfinite(aggregate_metrics(frame)["weekly_rank_ic"])
    }
    if not finite_control_scores:
        raise RuntimeError("no simple ranking control has finite validation rank IC")
    control_order = [
        "zero_return",
        "historical_mean",
        "persistence_1w",
        "reversal_1w",
        "momentum_4w",
        "ridge",
    ]
    strongest_control = _select_with_declared_tie_break(
        finite_control_scores,
        [name for name in control_order if name in finite_control_scores],
    )

    sensitivity: dict[str, dict[str, Any]] = {}
    val_sample = locked[locked["split"] == "validation"].iloc[:2]
    for name in ["canonical_independent_5ch", "canonical_mixing_5ch"]:
        x = np.stack(val_sample["x"]).astype(np.float32)
        sensitivity[name] = {}
        for seed in SEEDS:
            sensitivity[name][str(seed)] = model_sensitivity(
                trained[(name, seed, "daily_mse")], x, close_idx=3
            )
    if any(
        value["forecast_max_abs_delta"] != 0 or value["non_close_grad_l1"] != 0
        for value in sensitivity["canonical_independent_5ch"].values()
    ):
        raise RuntimeError(
            "channel-independent negative control was sensitive to non-Close inputs"
        )
    if any(
        value["forecast_max_abs_delta"] <= 0 or value["non_close_grad_l1"] <= 0
        for value in sensitivity["canonical_mixing_5ch"].values()
    ):
        raise RuntimeError("channel-mixing arm did not use non-Close inputs")
    if not all(
        np.isfinite(list(value.values())).all()
        for arm in sensitivity.values()
        for value in arm.values()
    ):
        raise RuntimeError("channel sensitivity contains non-finite values")

    selection = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "attempt_id": attempt_id,
        "test_labels_locked_during_selection": True,
        "selected_architecture": selected_architecture,
        "architecture_validation_rank_ic": architecture_validation_rank_ic,
        "selected_objective_arm": selected_objective_arm,
        "objective_validation_rank_ic": objective_validation_rank_ic,
        "strongest_simple_control": strongest_control,
        "control_validation_rank_ic": {
            name: aggregate_metrics(frame)["weekly_rank_ic"]
            for name, frame in validation_controls.items()
        },
        "validation_seed_results": validation_seed_metrics,
        "sensitivity": sensitivity,
        "config_sha256": manifest["config_sha256"],
        "selection_tie_break": config["selection_tie_break"],
    }
    json_dump(RESULTS / "selection.json", selection)
    pd.concat(
        [*sum(validation_frames.values(), []), *validation_controls.values()],
        ignore_index=True,
    ).to_csv(RESULTS / "validation_predictions.csv", index=False)
    unlock_receipt = {
        "config_sha256": manifest["config_sha256"],
        "source_sha256": manifest["source_sha256"],
        "selection_sha256": sha256_file(RESULTS / "selection.json"),
        "validation_predictions_sha256": sha256_file(
            RESULTS / "validation_predictions.csv"
        ),
    }
    unlock_path = RESULTS / "first_test_unlock.json"
    if unlock_path.exists():
        existing_unlock = json.loads(unlock_path.read_text())
        if existing_unlock != unlock_receipt:
            raise RuntimeError(
                "test was previously unlocked under a different config/source/selection; "
                "use a new experiment directory and holdout"
            )
    else:
        json_dump(unlock_path, unlock_receipt)
    print("Selection frozen. Unlocking test labels now.")

    panel = build_weekly_panel(prices, SPLITS, include_test_labels=True)
    locked_keys = (
        test_locked[["decision_date", "symbol"]]
        .astype(str)
        .to_records(index=False)
        .tolist()
    )
    test = panel[panel["split"] == "test"]
    open_keys = (
        test[["decision_date", "symbol"]].astype(str).to_records(index=False).tolist()
    )
    if locked_keys != open_keys or not np.isfinite(test["actual_weekly_return"]).all():
        raise RuntimeError("unlocked test panel differs from the frozen row set")

    prediction_frames: list[pd.DataFrame] = []
    aggregate_frames: dict[str, pd.DataFrame] = {}
    seed_results: dict[str, dict[str, Any]] = {}
    for name, architecture in ARCHITECTURES.items():
        test_x, _, _ = panel_arrays(panel, "test", architecture)
        frames: list[pd.DataFrame] = []
        seed_results[name] = {}
        for seed in SEEDS:
            frame = _prediction_frame(
                panel,
                "test",
                predict(trained[(name, seed, "daily_mse")], test_x),
                name,
                seed,
            )
            frames.append(frame)
            prediction_frames.append(frame)
            seed_results[name][str(seed)] = aggregate_metrics(frame)
        aggregate_frames[name] = _ensemble(frames, name, "test")

    test_x, _, _ = panel_arrays(panel, "test", selected)
    frames = []
    seed_results[listnet_name] = {}
    for seed in SEEDS:
        frame = _prediction_frame(
            panel,
            "test",
            predict(trained[(selected_architecture, seed, "listnet")], test_x),
            listnet_name,
            seed,
        )
        frames.append(frame)
        prediction_frames.append(frame)
        seed_results[listnet_name][str(seed)] = aggregate_metrics(frame)
    aggregate_frames[listnet_name] = _ensemble(frames, listnet_name, "test")

    controls = control_predictions(panel)
    control_frames: dict[str, pd.DataFrame] = {}
    for name in [
        "zero_return",
        "historical_mean",
        "majority_sign",
        "persistence_1w",
        "reversal_1w",
        "momentum_4w",
    ]:
        values = controls.loc[controls["split"] == "test", name].to_numpy()
        control_frames[name] = _prediction_frame(
            controls, "test", values, name, "control"
        )
    control_frames["ridge"] = _prediction_frame(
        panel, "test", ridge_predictions(panel), "ridge", "control"
    )
    prediction_frames.extend(control_frames.values())
    aggregate_frames.update(control_frames)

    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    all_predictions["decision_date"] = all_predictions["decision_date"].astype(str)
    all_predictions.to_csv(
        RESULTS / "predictions.csv", index=False, float_format="%.12g"
    )

    aggregate_results: dict[str, Any] = {}
    weekly_outputs: list[pd.DataFrame] = []
    for name, frame in aggregate_frames.items():
        metrics = aggregate_metrics(frame)
        intervals = bootstrap_intervals(frame, repetitions=2000)
        aggregate_results[name] = {"metrics": metrics, "ci95": intervals}
        weekly = weekly_statistics(frame)
        weekly["arm"] = name
        weekly_outputs.append(weekly)
    pd.concat(weekly_outputs, ignore_index=True).to_csv(
        RESULTS / "weekly_metrics.csv", index=False, float_format="%.12g"
    )

    daily_selected = aggregate_frames[selected_architecture]
    validation_selected = aggregate_frames[selected_objective_arm]
    comparisons = {
        "adapter_stride_fix_vs_legacy": paired_bootstrap_delta(
            aggregate_frames["stride_only_fixed"], aggregate_frames["legacy_effective"]
        ),
        "adapter_flatten_head_vs_stride_fix": paired_bootstrap_delta(
            aggregate_frames["canonical_independent_5ch"],
            aggregate_frames["stride_only_fixed"],
        ),
        "adapter_canonical_vs_legacy": paired_bootstrap_delta(
            aggregate_frames["canonical_independent_5ch"],
            aggregate_frames["legacy_effective"],
        ),
        "channel_mixing_vs_independent": paired_bootstrap_delta(
            aggregate_frames["canonical_mixing_5ch"],
            aggregate_frames["canonical_independent_5ch"],
        ),
        "listnet_vs_daily_mse": paired_bootstrap_delta(
            aggregate_frames[listnet_name], daily_selected
        ),
        "selected_vs_strongest_control": paired_bootstrap_delta(
            validation_selected, aggregate_frames[strongest_control]
        ),
    }
    verdicts = {}
    for name, comparison in comparisons.items():
        rank_interval = comparison["weekly_rank_ic"]["ci95"]
        top3_interval = comparison["top3_excess"]["ci95"]
        verdicts[name] = (
            "supported on both primary metrics"
            if rank_interval[0] > 0 and top3_interval[0] > 0
            else "no demonstrated improvement"
        )
    selected_ci = aggregate_results[selected_objective_arm]["ci95"]["weekly_rank_ic"]
    selected_top3_ci = aggregate_results[selected_objective_arm]["ci95"]["top3_excess"]
    advantage_ci = comparisons["selected_vs_strongest_control"]["weekly_rank_ic"][
        "ci95"
    ]
    advantage_top3_ci = comparisons["selected_vs_strongest_control"]["top3_excess"][
        "ci95"
    ]
    final_gate = (
        "demonstrated screening signal"
        if selected_ci[0] > 0
        and selected_top3_ci[0] > 0
        and advantage_ci[0] > 0
        and advantage_top3_ci[0] > 0
        else "no demonstrated signal"
    )
    evidence_status = "exploratory_after_review_reruns"

    metric_rows = []
    for name, result in aggregate_results.items():
        metric_rows.append(
            {
                "arm": name,
                **result["metrics"],
                "rank_ic_ci_low": result["ci95"]["weekly_rank_ic"][0],
                "rank_ic_ci_high": result["ci95"]["weekly_rank_ic"][1],
                "top3_excess_ci_low": result["ci95"]["top3_excess"][0],
                "top3_excess_ci_high": result["ci95"]["top3_excess"][1],
            }
        )
    pd.DataFrame(metric_rows).to_csv(
        RESULTS / "metrics.csv", index=False, float_format="%.12g"
    )

    active_model_directories = {
        MODELS / name / str(seed) for name in ARCHITECTURES for seed in SEEDS
    } | {MODELS / listnet_name / str(seed) for seed in SEEDS}
    active_model_files = sorted(
        path
        for directory in active_model_directories
        for path in directory.glob("*")
        if path.is_file()
    )
    manifest["model_artifact_sha256"] = {
        str(path.relative_to(BASE)): sha256_file(path) for path in active_model_files
    }
    all_model_files = {path for path in MODELS.rglob("*") if path.is_file()}
    manifest["stale_model_artifacts_excluded"] = sorted(
        str(path.relative_to(BASE))
        for path in all_model_files - set(active_model_files)
    )
    manifest["result_artifact_sha256"] = {
        str(path.relative_to(BASE)): sha256_file(path)
        for path in [
            RESULTS / "predictions.csv",
            RESULTS / "weekly_metrics.csv",
            RESULTS / "metrics.csv",
            RESULTS / "selection.json",
            RESULTS / "validation_predictions.csv",
            RESULTS / "first_test_unlock.json",
            BASE / "config.json",
        ]
    }
    manifest["run_log_note"] = (
        "run.log is append-only; prior failed/review attempts remain visible and are not part of the final-run integrity seal"
    )
    json_dump(RESULTS / "manifest.json", manifest)

    summary = {
        "status": "complete",
        "attempt_id": attempt_id,
        "evidence_status": evidence_status,
        "review_limitations": [
            "The 2024-2025 labels were opened by earlier review attempts before the final harness was frozen; final values are exploratory, not pristine confirmatory evidence.",
            "Pesaran-Timmermann is retained as an iid diagnostic only and is not used for inference because stock rows share decision-week shocks.",
            "The experiment uses a 12-stock survivor/current-universe diagnostic panel; it is not a production-universe backtest.",
        ],
        "manifest": manifest,
        "selection": {
            key: value
            for key, value in selection.items()
            if key != "validation_seed_results"
        },
        "seed_level_test_results": seed_results,
        "aggregate_test_results": aggregate_results,
        "arm_configs": {
            **architecture_manifest(),
            listnet_name: {
                **architecture_manifest()[selected_architecture],
                "objective": "listnet",
            },
        },
        "paired_comparisons": comparisons,
        "hypothesis_verdicts": verdicts,
        "final_gate": final_gate,
        "failures": failures,
        "failure_history_count": len(failure_history),
        "artifacts": {
            "predictions": "results/predictions.csv",
            "weekly_metrics": "results/weekly_metrics.csv",
            "metrics": "results/metrics.csv",
            "selection": "results/selection.json",
            "unlock_receipt": "results/first_test_unlock.json",
            "integrity": "results/integrity.json",
        },
    }
    json_dump(RESULTS / "summary.json", summary)
    json_dump(RESULTS / "failures.json", failures)
    json_dump(RESULTS / "failure_history.json", failure_history)
    integrity_paths = [
        BASE / "config.json",
        RESULTS / "manifest.json",
        RESULTS / "selection.json",
        RESULTS / "first_test_unlock.json",
        RESULTS / "summary.json",
        RESULTS / "failures.json",
        RESULTS / "failure_history.json",
        RESULTS / "predictions.csv",
        RESULTS / "weekly_metrics.csv",
        RESULTS / "metrics.csv",
        RESULTS / "validation_predictions.csv",
    ]
    json_dump(
        RESULTS / "integrity.json",
        {
            "attempt_id": attempt_id,
            "artifacts_sha256": {
                str(path.relative_to(BASE)): sha256_file(path)
                for path in integrity_paths
            },
        },
    )
    print(
        json.dumps(
            {
                "selected_architecture": selected_architecture,
                "selected_objective_arm": selected_objective_arm,
                "strongest_control": strongest_control,
                "verdicts": verdicts,
                "final_gate": final_gate,
                "metrics": {
                    name: value["metrics"] for name, value in aggregate_results.items()
                },
            },
            indent=2,
        )
    )
    return 0


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    lock_path = BASE / ".run.lock"
    with lock_path.open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "another corrected PatchTST suite process owns the experiment directory"
            ) from exc
        with LOG.open("a", buffering=1) as log_handle:
            tee_out, tee_err = Tee(sys.stdout, log_handle), Tee(sys.stderr, log_handle)
            with (
                contextlib.redirect_stdout(tee_out),
                contextlib.redirect_stderr(tee_err),
            ):
                print(f"process_id={os.getpid()}")
                try:
                    return run()
                except Exception as exc:
                    failure_path = RESULTS / "failures.json"
                    existing = (
                        json.loads(failure_path.read_text())
                        if failure_path.exists()
                        else []
                    )
                    suite_failure = {"stage": "suite", "error": repr(exc)}
                    if suite_failure not in existing:
                        existing.append(suite_failure)
                    json_dump(failure_path, existing)
                    history_path = RESULTS / "failure_history.json"
                    history = (
                        json.loads(history_path.read_text())
                        if history_path.exists()
                        else []
                    )
                    history.append(
                        {
                            "attempt_id": "suite-start-failure",
                            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                            **suite_failure,
                        }
                    )
                    json_dump(history_path, history)
                    print(f"SUITE FAILED: {exc!r}", file=sys.stderr)
                    raise


if __name__ == "__main__":
    raise SystemExit(main())

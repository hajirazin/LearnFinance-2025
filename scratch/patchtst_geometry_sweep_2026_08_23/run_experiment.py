#!/usr/bin/env python3
"""Run the frozen full-universe close-only PatchTST geometry sweep."""

from __future__ import annotations

import argparse
import subprocess
import time
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pandas as pd
import torch
from experiment_data import load_halal_new_universe_cache, load_or_download_prices
from geometry_metrics import (
    aggregate_metrics,
    build_causal_control_frames,
    paired_block_bootstrap,
    prediction_frame,
    research_clearance,
    weekly_metrics,
)
from geometry_panel import build_fold_panel, panel_arrays, panel_identity
from geometry_spec import (
    BOOTSTRAP_BLOCK_WEEKS,
    BOOTSTRAP_REPETITIONS,
    DATA_END_EXCLUSIVE,
    DATA_START,
    EVALUATION_FOLDS,
    PATCH_GEOMETRIES,
    SEEDS,
    TOP_K,
    frozen_configuration_manifest,
    json_dump,
    runtime_manifest,
    sha256_file,
)
from geometry_training import (
    TrainingJob,
    TrainingRunLock,
    cleanup_device,
    mps_smoke_result_is_sanitized,
    predict_weekly_log_returns,
    run_mps_determinism_smoke,
    train_geometry_seed,
    training_jobs,
)

BASE = Path(__file__).resolve().parent
REPOSITORY = BASE.parents[1]
DATA_DIR = BASE / "data"
MODEL_DIR = BASE / "models"
RESULTS_DIR = BASE / "results"
LOCK_PATH = BASE / ".training.lock"


class SequentialModelLedger:
    """Audit and enforce that at most one model is live in the runner."""

    def __init__(self) -> None:
        self.active_job: tuple[str, str, int] | None = None
        self.max_active_models = 0
        self.completed_jobs: list[tuple[str, str, int]] = []

    def begin(self, job: tuple[str, str, int]) -> None:
        if self.active_job is not None:
            raise RuntimeError(f"model job {self.active_job} is still active")
        self.active_job = job
        self.max_active_models = max(self.max_active_models, 1)

    def finish(self, job: tuple[str, str, int]) -> None:
        if self.active_job != job:
            raise RuntimeError(f"cannot finish inactive model job {job}")
        self.completed_jobs.append(job)
        self.active_job = None


def research_side_effects_manifest() -> dict[str, bool]:
    """Make the non-production contract machine-readable."""
    return {
        "production_current_pointers_touched": False,
        "artifacts_promoted": False,
        "trades_submitted": False,
        "temporal_workflows_triggered": False,
        "production_cache_written": False,
    }


def _git_value(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPOSITORY, text=True
    ).strip()


def confirmatory_unlock_payload(
    training_metadata: dict[str, dict[str, Any]],
    *,
    attempt_id: str,
    locked_non_evaluation_panel_sha256: str,
) -> dict[str, Any]:
    """Verify all nine confirmatory checkpoints before permitting label access."""
    prefix = "confirmatory_2026/"
    checkpoints = {
        key: value for key, value in training_metadata.items() if key.startswith(prefix)
    }
    if len(checkpoints) != 9:
        raise RuntimeError("confirmatory unlock requires exactly 9 checkpoints")
    expected = {
        f"confirmatory_2026/{geometry}/{seed}"
        for geometry in PATCH_GEOMETRIES
        for seed in SEEDS
    }
    if set(checkpoints) != expected:
        raise RuntimeError("confirmatory checkpoint keys do not match frozen jobs")
    verified: dict[str, Any] = {}
    for key, metadata in sorted(checkpoints.items()):
        path = Path(str(metadata["weights_path"]))
        if not path.exists() or sha256_file(path) != metadata["weights_sha256"]:
            raise RuntimeError(f"confirmatory checkpoint hash mismatch for {key}")
        verified[key] = {
            "weights_path": str(path),
            "weights_sha256": metadata["weights_sha256"],
            "created_at_utc": metadata["created_at_utc"],
        }
    return {
        "attempt_id": attempt_id,
        "unlocked_at_utc": datetime.now(UTC).isoformat(),
        "evaluation_labels_read_before_unlock": False,
        "checkpoint_count": len(verified),
        "locked_non_evaluation_panel_sha256": locked_non_evaluation_panel_sha256,
        "checkpoints": verified,
    }


def _ensemble(frames: list[pd.DataFrame], model: str) -> pd.DataFrame:
    stacked = pd.concat(frames, ignore_index=True)
    result = stacked.groupby(["fold", "decision_date", "symbol"], as_index=False).agg(
        actual_weekly_log_return=("actual_weekly_log_return", "first"),
        actual_weekly_return=("actual_weekly_return", "first"),
        predicted_weekly_log_return=("predicted_weekly_log_return", "mean"),
        predicted_weekly_return=("predicted_weekly_return", "mean"),
    )
    result["model"] = model
    result["seed"] = "ensemble"
    return result


def _fold_counts(panel: pd.DataFrame) -> dict[str, Any]:
    return {
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


def _checkpoint_model(job: TrainingJob) -> torch.nn.Module:
    path = MODEL_DIR / job.fold_name / job.geometry_name / str(job.seed) / "weights.pt"
    model = __import__("geometry_spec").build_patchtst_model(
        PATCH_GEOMETRIES[job.geometry_name]
    )
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model


def _run_one_fold(
    fold_name: str,
    prices: dict[str, pd.DataFrame],
    sessions: pd.DatetimeIndex,
    *,
    device: torch.device,
    max_epochs: int,
    patience: int,
    attempt_id: str,
    ledger: SequentialModelLedger,
    training_metadata: dict[str, dict[str, Any]],
) -> tuple[list[pd.DataFrame], dict[str, Any]]:
    fold = EVALUATION_FOLDS[fold_name]
    confirmatory = fold.evidence_kind == "confirmatory"
    panel = build_fold_panel(
        prices,
        sessions=sessions,
        fold=fold,
        include_evaluation_labels=not confirmatory,
    )
    train_x, train_y, _ = panel_arrays(panel, "train")
    validation_x, validation_y, validation_metadata = panel_arrays(panel, "validation")
    locked_non_evaluation_identity = panel_identity(panel, include_evaluation=False)
    frames: list[pd.DataFrame] = []
    seed_frames: dict[str, list[pd.DataFrame]] = {name: [] for name in PATCH_GEOMETRIES}

    for geometry_name, geometry in PATCH_GEOMETRIES.items():
        for seed in SEEDS:
            job_tuple = (fold_name, geometry_name, seed)
            ledger.begin(job_tuple)
            model: torch.nn.Module | None = None
            try:
                print(
                    f"TRAIN fold={fold_name} geometry={geometry_name} "
                    f"seed={seed} device={device}",
                    flush=True,
                )
                model_dir = MODEL_DIR / fold_name / geometry_name / str(seed)
                model, metadata = train_geometry_seed(
                    fold,
                    geometry,
                    seed,
                    train_x,
                    train_y,
                    validation_x,
                    validation_y,
                    validation_metadata,
                    model_dir=model_dir,
                    device=device,
                    max_epochs=max_epochs,
                    patience=patience,
                )
                metadata = {
                    **metadata,
                    "weights_path": str(model_dir / "weights.pt"),
                }
                training_metadata[f"{fold_name}/{geometry_name}/{seed}"] = metadata
                if not confirmatory:
                    evaluation_x, _, evaluation_metadata = panel_arrays(
                        panel, "evaluation"
                    )
                    predictions = predict_weekly_log_returns(
                        model.to(device), evaluation_x, device=device
                    )
                    frame = prediction_frame(
                        evaluation_metadata,
                        predictions,
                        model=geometry_name,
                        seed=str(seed),
                    )
                    frames.append(frame)
                    seed_frames[geometry_name].append(frame)
            finally:
                cleanup_device(model)
                ledger.finish(job_tuple)

    if confirmatory:
        unlock = confirmatory_unlock_payload(
            training_metadata,
            attempt_id=attempt_id,
            locked_non_evaluation_panel_sha256=locked_non_evaluation_identity,
        )
        json_dump(RESULTS_DIR / "confirmatory_unlock.json", unlock)
        panel = build_fold_panel(
            prices,
            sessions=sessions,
            fold=fold,
            include_evaluation_labels=True,
        )
        if (
            panel_identity(panel, include_evaluation=False)
            != locked_non_evaluation_identity
        ):
            raise RuntimeError("non-evaluation panel changed at confirmatory unlock")
        evaluation_x, _, evaluation_metadata = panel_arrays(panel, "evaluation")
        for geometry_name in PATCH_GEOMETRIES:
            for seed in SEEDS:
                job = TrainingJob(fold_name, geometry_name, seed)
                job_tuple = tuple(job)
                ledger.begin(job_tuple)
                model = None
                try:
                    model = _checkpoint_model(job).to(device)
                    predictions = predict_weekly_log_returns(
                        model, evaluation_x, device=device
                    )
                    frame = prediction_frame(
                        evaluation_metadata,
                        predictions,
                        model=geometry_name,
                        seed=str(seed),
                    )
                    frames.append(frame)
                    seed_frames[geometry_name].append(frame)
                finally:
                    cleanup_device(model)
                    ledger.finish(job_tuple)

    for geometry_name, values in seed_frames.items():
        frames.append(_ensemble(values, geometry_name))
    controls = build_causal_control_frames(panel)
    frames.extend(controls.values())
    return frames, {
        "counts": _fold_counts(panel),
        "exclusion_counts": panel.attrs["exclusion_counts"],
        "per_symbol_counts": panel.attrs["per_symbol_counts"],
        "non_evaluation_panel_sha256": locked_non_evaluation_identity,
    }


def _evidence_frame(
    frames: list[pd.DataFrame], model: str, evidence_kind: str
) -> pd.DataFrame:
    selected = [
        frame
        for frame in frames
        if frame["model"].iloc[0] == model
        and frame["seed"].iloc[0] in ("ensemble", "control")
        and EVALUATION_FOLDS[str(frame["fold"].iloc[0])].evidence_kind == evidence_kind
    ]
    if not selected:
        raise RuntimeError(f"no {evidence_kind} frames for {model}")
    return pd.concat(selected, ignore_index=True)


def _comparisons(frames: list[pd.DataFrame]) -> tuple[dict[str, Any], dict[str, Any]]:
    gates = ("causal_historical_mean", "ridge")
    arm_gate: dict[str, Any] = {}
    for geometry_name in PATCH_GEOMETRIES:
        arm_gate[geometry_name] = {}
        for evidence_kind in ("development", "confirmatory"):
            challenger = _evidence_frame(frames, geometry_name, evidence_kind)
            arm_gate[geometry_name][evidence_kind] = {
                gate: paired_block_bootstrap(
                    challenger,
                    _evidence_frame(frames, gate, evidence_kind),
                    seed=20260823,
                    repetitions=BOOTSTRAP_REPETITIONS,
                    block_weeks=BOOTSTRAP_BLOCK_WEEKS,
                    top_k=TOP_K,
                )
                for gate in gates
            }
    arm_pairs: dict[str, Any] = {}
    for evidence_kind in ("development", "confirmatory"):
        arm_pairs[evidence_kind] = {}
        for challenger_name, reference_name in combinations(PATCH_GEOMETRIES, 2):
            arm_pairs[evidence_kind][f"{challenger_name}_minus_{reference_name}"] = (
                paired_block_bootstrap(
                    _evidence_frame(frames, challenger_name, evidence_kind),
                    _evidence_frame(frames, reference_name, evidence_kind),
                    seed=20260823,
                    repetitions=BOOTSTRAP_REPETITIONS,
                    block_weeks=BOOTSTRAP_BLOCK_WEEKS,
                    top_k=TOP_K,
                )
            )
    return arm_gate, arm_pairs


def _source_hashes() -> dict[str, str]:
    return {
        path.name: sha256_file(path)
        for path in BASE.glob("*.py")
        if path.name != "test_experiment.py"
    }


def run(
    *,
    device_name: str,
    universe_cache: Path,
    max_epochs: int,
    patience: int,
    smoke_only: bool,
) -> None:
    """Execute acquisition, sequential training, locked evaluation, and audit."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    attempt_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    started = time.perf_counter()
    json_dump(
        RESULTS_DIR / "status.json",
        {"status": "in_progress", "attempt_id": attempt_id},
    )
    try:
        with TrainingRunLock(LOCK_PATH):
            smoke = run_mps_determinism_smoke()
            if not mps_smoke_result_is_sanitized(smoke) or not smoke["passed"]:
                raise RuntimeError(f"deterministic MPS smoke failed: {smoke}")
            json_dump(RESULTS_DIR / "mps_smoke.json", smoke)
            if smoke_only:
                json_dump(
                    RESULTS_DIR / "status.json",
                    {"status": "smoke_complete", "attempt_id": attempt_id},
                )
                return

            symbols, universe_manifest = load_halal_new_universe_cache(universe_cache)
            prices, price_manifest = load_or_download_prices(
                symbols,
                data_dir=DATA_DIR,
                start_date=DATA_START,
                end_date=DATA_END_EXCLUSIVE,
            )
            calendar = xcals.get_calendar("XNYS")
            sessions = calendar.sessions_in_range(DATA_START, DATA_END_EXCLUSIVE)
            device = torch.device(device_name)
            ledger = SequentialModelLedger()
            training_metadata: dict[str, dict[str, Any]] = {}
            frames: list[pd.DataFrame] = []
            panel_manifests: dict[str, Any] = {}
            for fold_name in EVALUATION_FOLDS:
                fold_frames, panel_manifest = _run_one_fold(
                    fold_name,
                    prices,
                    sessions,
                    device=device,
                    max_epochs=max_epochs,
                    patience=patience,
                    attempt_id=attempt_id,
                    ledger=ledger,
                    training_metadata=training_metadata,
                )
                frames.extend(fold_frames)
                panel_manifests[fold_name] = panel_manifest

            metrics: dict[str, Any] = {}
            weekly_frames: list[pd.DataFrame] = []
            for frame in frames:
                key = "/".join(
                    [
                        str(frame["fold"].iloc[0]),
                        str(frame["model"].iloc[0]),
                        str(frame["seed"].iloc[0]),
                    ]
                )
                metrics[key] = aggregate_metrics(frame, top_k=TOP_K)
                weekly = weekly_metrics(frame, top_k=TOP_K)
                weekly["model"] = frame["model"].iloc[0]
                weekly["seed"] = frame["seed"].iloc[0]
                weekly_frames.append(weekly)
            arm_gate, arm_pairs = _comparisons(frames)
            clearance: dict[str, Any] = {}
            for geometry_name in PATCH_GEOMETRIES:
                fold_metrics = {
                    fold_name: metrics[f"{fold_name}/{geometry_name}/ensemble"]
                    for fold_name in EVALUATION_FOLDS
                }
                clearance[geometry_name] = research_clearance(
                    fold_metrics, arm_gate[geometry_name]
                )

            predictions_path = RESULTS_DIR / "predictions.csv"
            weekly_path = RESULTS_DIR / "weekly_metrics.csv"
            pd.concat(frames, ignore_index=True).to_csv(predictions_path, index=False)
            pd.concat(weekly_frames, ignore_index=True).to_csv(weekly_path, index=False)
            unlock_path = RESULTS_DIR / "confirmatory_unlock.json"
            manifest = {
                "status": "complete",
                "attempt_id": attempt_id,
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "runtime_seconds": time.perf_counter() - started,
                "git_head": _git_value("rev-parse", "HEAD"),
                "git_branch": _git_value("branch", "--show-current"),
                "git_status_short": _git_value("status", "--short"),
                "runtime": runtime_manifest(),
                "device_used": device_name,
                "mps_smoke": smoke,
                "frozen_configuration": frozen_configuration_manifest(),
                "universe": universe_manifest,
                "price_data": price_manifest,
                "requested_symbol_count": len(symbols),
                "downloaded_symbol_count": len(prices),
                "panel_manifests": panel_manifests,
                "training": training_metadata,
                "sequential_execution": {
                    "declared_training_jobs": len(training_jobs()),
                    "completed_training_jobs": 27,
                    "completed_model_lifecycles": len(ledger.completed_jobs),
                    "max_active_models": ledger.max_active_models,
                    "execution_order": ledger.completed_jobs,
                },
                "metrics": metrics,
                "paired_block_bootstrap": {
                    "repetitions": BOOTSTRAP_REPETITIONS,
                    "block_weeks": BOOTSTRAP_BLOCK_WEEKS,
                    "unit": "decision_week",
                    "arm_vs_gates": arm_gate,
                    "arm_vs_arm": arm_pairs,
                },
                "research_clearance": clearance,
                "source_sha256": _source_hashes(),
                "artifact_sha256": {
                    "predictions.csv": sha256_file(predictions_path),
                    "weekly_metrics.csv": sha256_file(weekly_path),
                    "confirmatory_unlock.json": sha256_file(unlock_path),
                    "mps_smoke.json": sha256_file(RESULTS_DIR / "mps_smoke.json"),
                },
                **research_side_effects_manifest(),
                "limitations": [
                    "Current August 2026 halal_new membership is used historically, creating survivorship and membership bias.",
                    "Yahoo adjusted prices are revision-prone and not institutional point-in-time data.",
                    "2024-2025 is development evidence already opened before this geometry preregistration.",
                    "The 2026 confirmatory path is partial through the five sessions ending 2026-08-21.",
                    "Top-15 diagnostics are gross signals without sticky selection, HRP, costs, or execution.",
                    "Three seeds and one historical path leave material model and regime uncertainty.",
                ],
            }
            json_dump(RESULTS_DIR / "manifest.json", manifest)
            json_dump(
                RESULTS_DIR / "status.json",
                {"status": "complete", "attempt_id": attempt_id},
            )
    except Exception as error:
        json_dump(
            RESULTS_DIR / "status.json",
            {
                "status": "failed",
                "attempt_id": attempt_id,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["mps"], default="mps")
    parser.add_argument("--universe-cache", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--smoke-only", action="store_true")
    arguments = parser.parse_args()
    run(
        device_name=arguments.device,
        universe_cache=arguments.universe_cache,
        max_epochs=arguments.max_epochs,
        patience=arguments.patience,
        smoke_only=arguments.smoke_only,
    )


if __name__ == "__main__":
    main()

"""Disposable, offline orchestration checks; no model training is run."""

import hashlib
import json

import pytest
from run_all import run_pipeline


def prepared_root(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    content = b"disposable frozen inputs"
    for filename in ("panel.pkl", "adjusted_ohlcv.parquet"):
        (evidence / filename).write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    (evidence / "manifest.json").write_text(
        json.dumps({"panel_sha256": digest, "price_sha256": digest})
    )
    return tmp_path


def test_one_shot_finishes_every_stage_and_records_completion(tmp_path):
    root = prepared_root(tmp_path)
    stages = []

    def successful_stage(stage, destination):
        stages.append(stage)
        if stage == "verify_results":
            output = destination / "results_50000"
            output.mkdir()
            (output / "report.md").write_text("Completed report")
            (output / "verification.json").write_text(
                json.dumps(
                    {
                        "matched_training_runs": 12,
                        "execution_sensitivity_evaluations": 48,
                    }
                )
            )

    run_pipeline(root, stage_runner=successful_stage)
    assert stages == [
        "run_experiment",
        "execution_sensitivity",
        "summarize",
        "verify_results",
    ]
    assert json.loads((root / "status.json").read_text())["phase"] == "complete"


def test_failed_stage_stops_pipeline_and_reports_failure(tmp_path):
    root = prepared_root(tmp_path)
    stages = []

    def failed_stage(stage, destination):
        stages.append(stage)
        if stage == "execution_sensitivity":
            raise RuntimeError("evaluation failed")

    with pytest.raises(RuntimeError, match="evaluation failed"):
        run_pipeline(root, stage_runner=failed_stage)
    assert stages == ["run_experiment", "execution_sensitivity"]
    status = json.loads((root / "status.json").read_text())
    assert status["phase"] == "failed"
    assert "evaluation failed" in status["error"]


def test_changed_frozen_inputs_never_start_training(tmp_path):
    root = prepared_root(tmp_path)
    (root / "evidence/panel.pkl").write_bytes(b"changed")
    stages = []
    with pytest.raises(ValueError, match="Frozen input changed"):
        run_pipeline(root, stage_runner=lambda stage, destination: stages.append(stage))
    assert stages == []

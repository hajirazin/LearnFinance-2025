"""Tests for ``evaluate_forecaster_artifact_health``.

Each guardrail has a happy-path assertion + a dedicated failure test.
The combined-failure test proves that ``failure_reasons`` accumulates
multiple problems in one health-check call (we don't short-circuit on
the first failure -- the operator wants to fix everything in one go).
"""

import math
from pathlib import Path

import pytest

from brain_api.core.training_utils import (
    _FORECASTER_REQUIRED_FILES,
    evaluate_forecaster_artifact_health,
)


@pytest.fixture
def healthy_artifact_dir(tmp_path: Path) -> Path:
    """Materialize the four files a healthy forecaster run writes."""
    for filename in _FORECASTER_REQUIRED_FILES:
        (tmp_path / filename).write_bytes(b"non-empty")
    return tmp_path


def _healthy_metrics() -> dict[str, float]:
    """Plausible LSTM val_loss range; numbers don't matter as long as
    they're finite and positive."""
    return {
        "train_loss": 0.001,
        "val_loss": 0.002,
        "baseline_loss": 0.003,
    }


def test_happy_path_promotes(healthy_artifact_dir: Path):
    health = evaluate_forecaster_artifact_health(
        artifact_dir=healthy_artifact_dir, **_healthy_metrics()
    )
    assert health.is_healthy is True
    assert health.failure_reasons == []


def test_nan_val_loss_rejects(healthy_artifact_dir: Path):
    health = evaluate_forecaster_artifact_health(
        train_loss=0.001,
        val_loss=float("nan"),
        baseline_loss=0.003,
        artifact_dir=healthy_artifact_dir,
    )
    assert health.is_healthy is False
    assert "val_loss is not finite" in health.failure_reasons


def test_inf_val_loss_rejects(healthy_artifact_dir: Path):
    health = evaluate_forecaster_artifact_health(
        train_loss=0.001,
        val_loss=math.inf,
        baseline_loss=0.003,
        artifact_dir=healthy_artifact_dir,
    )
    assert "val_loss is not finite" in health.failure_reasons


def test_zero_val_loss_rejects(healthy_artifact_dir: Path):
    health = evaluate_forecaster_artifact_health(
        train_loss=0.001,
        val_loss=0.0,
        baseline_loss=0.003,
        artifact_dir=healthy_artifact_dir,
    )
    assert any("val_loss must be > 0" in r for r in health.failure_reasons)


def test_negative_val_loss_rejects(healthy_artifact_dir: Path):
    health = evaluate_forecaster_artifact_health(
        train_loss=0.001,
        val_loss=-1.0,
        baseline_loss=0.003,
        artifact_dir=healthy_artifact_dir,
    )
    assert any("val_loss must be > 0" in r for r in health.failure_reasons)


def test_nan_train_loss_rejects(healthy_artifact_dir: Path):
    health = evaluate_forecaster_artifact_health(
        train_loss=float("nan"),
        val_loss=0.002,
        baseline_loss=0.003,
        artifact_dir=healthy_artifact_dir,
    )
    assert "train_loss is not finite" in health.failure_reasons


def test_negative_train_loss_rejects(healthy_artifact_dir: Path):
    health = evaluate_forecaster_artifact_health(
        train_loss=-0.001,
        val_loss=0.002,
        baseline_loss=0.003,
        artifact_dir=healthy_artifact_dir,
    )
    assert any("train_loss must be > 0" in r for r in health.failure_reasons)


def test_nan_baseline_loss_rejects(healthy_artifact_dir: Path):
    health = evaluate_forecaster_artifact_health(
        train_loss=0.001,
        val_loss=0.002,
        baseline_loss=float("nan"),
        artifact_dir=healthy_artifact_dir,
    )
    assert "baseline_loss is not finite" in health.failure_reasons


def test_negative_baseline_loss_rejects(healthy_artifact_dir: Path):
    health = evaluate_forecaster_artifact_health(
        train_loss=0.001,
        val_loss=0.002,
        baseline_loss=-0.003,
        artifact_dir=healthy_artifact_dir,
    )
    assert any("baseline_loss must be > 0" in r for r in health.failure_reasons)


def test_snapshot_metrics_only_skips_baseline_and_files():
    """Snapshot persist has no baseline and checks metrics before write."""
    health = evaluate_forecaster_artifact_health(
        train_loss=0.001,
        val_loss=0.002,
        baseline_loss=None,
        artifact_dir=None,
    )
    assert health.is_healthy is True
    assert health.failure_reasons == []


def test_snapshot_metrics_only_rejects_nan_val_loss():
    health = evaluate_forecaster_artifact_health(
        train_loss=0.001,
        val_loss=float("nan"),
        baseline_loss=None,
        artifact_dir=None,
    )
    assert health.is_healthy is False
    assert "val_loss is not finite" in health.failure_reasons
    assert not any("baseline_loss" in r for r in health.failure_reasons)
    assert not any("missing or zero bytes" in r for r in health.failure_reasons)


@pytest.mark.parametrize("missing_file", _FORECASTER_REQUIRED_FILES)
def test_missing_file_rejects(tmp_path: Path, missing_file: str):
    """Each of the four artifact files must exist with non-zero size."""
    for filename in _FORECASTER_REQUIRED_FILES:
        if filename == missing_file:
            continue
        (tmp_path / filename).write_bytes(b"x")

    health = evaluate_forecaster_artifact_health(
        artifact_dir=tmp_path, **_healthy_metrics()
    )
    assert health.is_healthy is False
    assert f"{missing_file} missing or zero bytes" in health.failure_reasons


@pytest.mark.parametrize("zero_byte_file", _FORECASTER_REQUIRED_FILES)
def test_zero_byte_file_rejects(tmp_path: Path, zero_byte_file: str):
    """Zero-byte file is treated identically to a missing file -- this
    catches torch save corruption that surfaces as a 0-byte weights.pt."""
    for filename in _FORECASTER_REQUIRED_FILES:
        contents = b"" if filename == zero_byte_file else b"x"
        (tmp_path / filename).write_bytes(contents)

    health = evaluate_forecaster_artifact_health(
        artifact_dir=tmp_path, **_healthy_metrics()
    )
    assert f"{zero_byte_file} missing or zero bytes" in health.failure_reasons


def test_combined_failures_accumulate(tmp_path: Path):
    """Multiple guardrail failures must all show up in the same health
    check so the operator-facing email lists them all in one go."""
    # Materialize only 2 of the 4 required files
    (tmp_path / "weights.pt").write_bytes(b"x")
    (tmp_path / "config.json").write_bytes(b"x")

    health = evaluate_forecaster_artifact_health(
        train_loss=float("nan"),
        val_loss=-1.0,
        baseline_loss=0.003,
        artifact_dir=tmp_path,
    )
    assert health.is_healthy is False
    reasons = health.failure_reasons
    assert "train_loss is not finite" in reasons
    assert any("val_loss must be > 0" in r for r in reasons)
    assert "feature_scaler.pkl missing or zero bytes" in reasons
    assert "metadata.json missing or zero bytes" in reasons


def test_is_accelerator_out_of_memory_detects_python_cuda_and_mps() -> None:
    import torch

    from brain_api.core.training_utils import is_accelerator_out_of_memory

    cpu = torch.device("cpu")
    mps = torch.device("mps")
    assert is_accelerator_out_of_memory(MemoryError("oom"), cpu)
    assert is_accelerator_out_of_memory(RuntimeError("MPS backend out of memory"), mps)
    assert not is_accelerator_out_of_memory(RuntimeError("out of memory"), cpu)
    assert not is_accelerator_out_of_memory(RuntimeError("shape mismatch"), mps)
    cuda_oom = getattr(torch.cuda, "OutOfMemoryError", None)
    if cuda_oom is not None:
        assert is_accelerator_out_of_memory(cuda_oom("cuda oom"), torch.device("cuda"))

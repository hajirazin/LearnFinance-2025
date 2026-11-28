"""Exercise checkpoint-to-report evaluation entirely with disposable evidence."""

from datetime import date, timedelta

import numpy as np
import torch
from reporting import evaluate
from test_training import synthetic_panel, tiny_config
from training import fit_arm

from brain_api.core.patchtst.weekly_data import WeeklyPatchTSTData


def test_complete_paired_evaluation(monkeypatch, tmp_path):
    torch.set_num_threads(1)
    config, fitting = tiny_config(), synthetic_panel()
    for arm in ("mse", "huber"):
        fit_arm(fitting, config, arm, torch.device("cpu"), tmp_path / arm)
    anchors = np.repeat([date(2025, 1, 3) + timedelta(weeks=i) for i in range(20)], 30)
    rng = np.random.default_rng(81)
    x = rng.normal(size=(600, 255, 7)).astype(np.float32)
    panel = WeeklyPatchTSTData(
        x,
        (0.02 * x[:, -1, 0]).astype(np.float32),
        anchors,
        anchors + timedelta(days=7),
        np.tile([f"S{i:02d}" for i in range(30)], 20),
    )
    monkeypatch.setattr("reporting.ROOT", tmp_path)
    result = evaluate(panel, config, torch.device("cpu"), tmp_path, {"folds": {}})
    assert result["weeks"] == 20
    assert result["outcomes"] == 600
    assert len(result["paired_ic_intervals"]) == 3
    assert (tmp_path / "REPORT.md").is_file()
    assert (tmp_path / "predictions.csv").is_file()
    # The comparison must not accept a checkpoint different from its fit record.
    with (tmp_path / "huber/model.pt").open("ab") as handle:
        handle.write(b"changed")
    import pytest

    with pytest.raises(ValueError, match="Checkpoint changed"):
        evaluate(panel, config, torch.device("cpu"), tmp_path, {"folds": {}})

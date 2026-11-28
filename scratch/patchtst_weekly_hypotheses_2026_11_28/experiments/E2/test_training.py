"""Check loss mathematics and parity with the real weekly MSE training path."""

from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pytest
import torch
from training import fit_arm, training_objective

from brain_api.core.patchtst.config import US_WEEKLY_CONFIG
from brain_api.core.patchtst.weekly_data import WeeklyPatchTSTData
from brain_api.core.patchtst.weekly_training import (
    fit_weekly_patchtst,
    weekly_fold_indices,
)


def test_huber_cutoff_in_return_units_and_gradient_cap():
    # 2% target std makes the raw 5pp cutoff equal 2.5 scaled units.
    residual = torch.tensor([0.5, 2.5, 5.0, -5.0], requires_grad=True)
    actual = torch.zeros(4)
    loss = training_objective(residual, actual, "huber", 0.02, reduction="none")
    torch.testing.assert_close(loss, torch.tensor([0.25, 6.25, 18.75, 18.75]))
    loss.sum().backward()
    torch.testing.assert_close(residual.grad, torch.tensor([1.0, 5.0, 5.0, -5.0]))
    assert training_objective(residual, actual, "mse", 0.02).item() == 14.125


@pytest.mark.parametrize("scale", [0.0, -0.1, float("nan")])
def test_invalid_scale_rejected(scale):
    with pytest.raises(ValueError):
        training_objective(torch.ones(1), torch.zeros(1), "huber", scale)


def synthetic_panel():
    dates = np.repeat([date(2018, 1, 5) + timedelta(weeks=i) for i in range(70)], 3)
    rng = np.random.default_rng(91)
    x = rng.normal(size=(len(dates), 255, 7)).astype(np.float32)
    y = (0.03 * x[:, -1, 0] + 0.01).astype(np.float32)
    return WeeklyPatchTSTData(
        x, y, dates, dates + timedelta(days=7), np.tile(["A", "B", "C"], 70)
    )


def tiny_config():
    return replace(
        US_WEEKLY_CONFIG,
        d_model=4,
        num_attention_heads=1,
        num_hidden_layers=1,
        ffn_dim=8,
        patch_length=128,
        stride=127,
        batch_size=32,
        epochs=2,
        early_stopping_patience=1,
    )


def test_purge_and_real_mse_trainer_parity(monkeypatch, tmp_path):
    torch.set_num_threads(1)
    monkeypatch.setattr(
        "brain_api.core.patchtst.weekly_training.get_device",
        lambda: torch.device("cpu"),
    )
    panel, config = synthetic_panel(), tiny_config()
    train, validation = weekly_fold_indices(panel, config.validation_split)
    assert max(panel.target_dates[train]) < min(panel.anchor_dates[validation])
    production = fit_weekly_patchtst(panel, config)
    scratch = fit_arm(panel, config, "mse", torch.device("cpu"), tmp_path / "mse")
    assert scratch["best_epoch"] == production.best_epoch
    assert scratch["stopped_epoch"] == production.stopped_epoch
    assert scratch["validation_mse"] == pytest.approx(production.val_loss, abs=1e-12)
    checkpoint = torch.load(tmp_path / "mse/model.pt", weights_only=True)
    for key, value in production.model.state_dict().items():
        torch.testing.assert_close(checkpoint[key], value, rtol=0, atol=0)


def test_huber_runs_and_persists_learning_history(tmp_path):
    torch.set_num_threads(1)
    result = fit_arm(
        synthetic_panel(), tiny_config(), "huber", torch.device("cpu"), tmp_path
    )
    assert result["stopped_epoch"] == len(result["history"])
    assert np.isfinite(result["validation_mse"])
    assert all(0 <= row["gradient_clipped_fraction"] <= 1 for row in result["history"])
    assert (tmp_path / "model.pt").is_file()

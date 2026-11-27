"""Progress and cancellation contracts for the production weekly trainer."""

import logging
from dataclasses import replace
from datetime import date, timedelta
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from brain_api.core.patchtst.config import INDIA_WEEKLY_CONFIG, US_WEEKLY_CONFIG
from brain_api.core.patchtst.weekly_data import WeeklyPatchTSTData
from brain_api.core.patchtst.weekly_training import fit_weekly_patchtst
from brain_api.core.training_utils import TrainingCancelledError


def synthetic_weekly_data():
    rng = np.random.default_rng(7)
    anchors = np.repeat([date(2020, 1, 3) + timedelta(weeks=i) for i in range(80)], 3)
    return WeeklyPatchTSTData(
        rng.normal(size=(len(anchors), 255, 6)).astype(np.float32),
        rng.normal(0, 0.03, size=len(anchors)).astype(np.float32),
        anchors,
        np.array([day + timedelta(days=7) for day in anchors]),
        np.tile(["AAA.NS", "BBB.NS", "CCC.NS"], 80),
    )


@pytest.fixture
def tiny_regression_model(monkeypatch):
    class TinyRegressor(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.linear = torch.nn.Linear(config.num_input_channels, 1)

        def forward(self, past_values):
            return SimpleNamespace(
                regression_outputs=self.linear(past_values.mean(dim=1))
            )

    monkeypatch.setattr(
        "brain_api.core.patchtst.weekly_training.create_patchtst_model", TinyRegressor
    )
    monkeypatch.setattr(
        "brain_api.core.patchtst.weekly_training.get_device",
        lambda: torch.device("cpu"),
    )


def test_both_markets_restore_patience_fifteen():
    assert US_WEEKLY_CONFIG.early_stopping_patience == 15
    assert INDIA_WEEKLY_CONFIG.early_stopping_patience == 15


def test_epoch_metrics_and_early_stop_are_logged(tiny_regression_model, caplog):
    config = replace(
        INDIA_WEEKLY_CONFIG, epochs=3, early_stopping_patience=1, learning_rate=0.0
    )
    events = []
    caplog.set_level(logging.INFO)
    result = fit_weekly_patchtst(
        synthetic_weekly_data(), config, on_progress=events.append
    )
    assert result.best_epoch == 1
    assert result.stopped_epoch == 2
    completed = [event for event in events if "val_loss" in event]
    assert [event["epoch"] for event in completed] == [1, 2]
    for event in completed:
        assert event["weekly_rmse_pp"] == pytest.approx(
            100 * np.sqrt(event["val_loss"])
        )
        assert event["patience_limit"] == 1
        assert event["learning_rate"] == 0.0
    assert "epoch=1/3" in caplog.text
    assert "early stopping" in caplog.text.lower()
    assert "best_epoch=1" in caplog.text


def test_progress_callback_can_cancel_next_epoch(tiny_regression_model):
    shutdown = Event()

    def cancel_after_first_epoch(event):
        if "val_loss" in event:
            shutdown.set()

    with pytest.raises(TrainingCancelledError):
        fit_weekly_patchtst(
            synthetic_weekly_data(),
            replace(INDIA_WEEKLY_CONFIG, epochs=3),
            shutdown_event=shutdown,
            on_progress=cancel_after_first_epoch,
        )


def test_cancelled_inputs_do_not_start_price_loading():
    from brain_api.core.patchtst.weekly_data import load_weekly_training_inputs

    shutdown = Event()
    shutdown.set()
    loader = Mock(side_effect=AssertionError("cancelled job loaded prices"))
    with pytest.raises(TrainingCancelledError):
        load_weekly_training_inputs(
            ["AAA.NS"],
            date(2023, 1, 1),
            date(2024, 11, 29),
            INDIA_WEEKLY_CONFIG,
            loader,
            shutdown_event=shutdown,
        )
    loader.assert_not_called()

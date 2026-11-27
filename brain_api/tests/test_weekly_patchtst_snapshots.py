"""Annual snapshot orchestration shares progress and cancellation with its fits."""

from datetime import date
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from brain_api.core.patchtst.config import US_WEEKLY_CONFIG
from brain_api.core.training_utils import TrainingCancelledError
from brain_api.services import patchtst_snapshots as snapshots


def test_cancelled_snapshot_job_does_not_load_prices(tmp_path):
    event = Event()
    event.set()
    loader = Mock(side_effect=AssertionError("cancelled snapshot loaded prices"))
    with pytest.raises(TrainingCancelledError):
        snapshots.backfill_weekly_patchtst_snapshots(
            SimpleNamespace(base_path=tmp_path),
            ["AAA"],
            US_WEEKLY_CONFIG,
            date(2024, 1, 1),
            date(2026, 11, 20),
            loader,
            log_prefix="test",
            shutdown_event=event,
        )
    loader.assert_not_called()


def test_snapshot_fit_receives_event_and_reports_year(tmp_path, monkeypatch):
    cutoff = date(2025, 12, 31)
    event, progress = Event(), []
    storage = SimpleNamespace(
        base_path=tmp_path,
        forecaster_type="patchtst_halal_new",
        _get_hf_repo=lambda: None,
    )
    monkeypatch.setattr(snapshots, "annual_snapshot_cutoffs", lambda *args: [cutoff])
    monkeypatch.setattr(snapshots, "annual_snapshot_parity", lambda *args: "train")
    inputs = SimpleNamespace(labeled=Mock(return_value="synthetic dataset"))
    monkeypatch.setattr(
        snapshots, "load_weekly_training_inputs", Mock(return_value=inputs)
    )

    def fit(data, config, *, shutdown_event, on_progress):
        assert shutdown_event is event
        assert data == "synthetic dataset"
        on_progress(
            {"phase": "training_weekly_regression", "epoch": 1, "total_epochs": 60}
        )
        return SimpleNamespace(
            model=object(),
            feature_scaler=object(),
            train_loss=0.01,
            val_loss=0.02,
            best_epoch=1,
            stopped_epoch=1,
            evidence={"latest_label_end": "2025-12-26", "weekly_rmse_pp": 14.14},
        )

    monkeypatch.setattr(
        snapshots,
        "persist_forecaster_snapshot",
        lambda **kwargs: SimpleNamespace(is_canonical=True),
    )
    first = snapshots.backfill_weekly_patchtst_snapshots(
        storage,
        ["AAA"],
        US_WEEKLY_CONFIG,
        date(2024, 1, 1),
        date(2026, 11, 20),
        Mock(),
        log_prefix="test",
        shutdown_event=event,
        on_progress=progress.append,
        trainer=fit,
    )
    assert first == cutoff
    fitting = next(item for item in progress if item.get("epoch") == 1)
    assert fitting["phase"] == "training_annual_snapshots"
    assert fitting["snapshot_cutoff"] == "2025-12-31"
    assert fitting["snapshot_index"] == fitting["snapshot_total"] == 1
    assert inputs.labeled.call_args.kwargs["shutdown_event"] is event

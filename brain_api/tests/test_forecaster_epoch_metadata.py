"""Epoch counters on forecaster TrainingResult and metadata.json."""

from datetime import date

import numpy as np
from sklearn.preprocessing import StandardScaler

from brain_api.core.lstm.config import LSTMConfig
from brain_api.core.lstm.training import train_model_pytorch as train_lstm
from brain_api.core.patchtst.config import INDIA_WEEKLY_CONFIG
from brain_api.core.patchtst.weekly_training import fit_weekly_patchtst
from brain_api.storage.metadata import create_training_metadata
from tests.test_weekly_patchtst_progress import synthetic_weekly_data


def test_create_training_metadata_includes_epoch_counters() -> None:
    metadata = create_training_metadata(
        model_type="lstm_halal_new",
        version="v2026-08-15-aaaaaaaaaaaa",
        data_window_start="2016-01-01",
        data_window_end="2025-12-31",
        symbols=["AAPL"],
        config_dict={"epochs": 100},
        train_loss=0.01,
        val_loss=0.02,
        baseline_loss=0.03,
        best_epoch=2,
        stopped_epoch=17,
        promoted=True,
        prior_version=None,
    )
    assert metadata["metrics"]["best_epoch"] == 2
    assert metadata["metrics"]["stopped_epoch"] == 17
    assert metadata["metrics"]["train_loss"] == 0.01
    assert "val_rank_ic" not in metadata["metrics"]


def test_lstm_empty_data_records_zero_epochs() -> None:
    result = train_lstm(
        np.zeros((0, 60, 5), dtype=np.float32),
        np.zeros((0, 5), dtype=np.float32),
        StandardScaler(),
        LSTMConfig(),
    )
    assert result.best_epoch == 0
    assert result.stopped_epoch == 0


def test_lstm_tiny_train_records_epoch_counters() -> None:
    rng = np.random.default_rng(0)
    n_samples = 20
    config = LSTMConfig(
        epochs=2,
        early_stopping_patience=10,
        batch_size=8,
        hidden_size=16,
    )
    result = train_lstm(
        rng.normal(size=(n_samples, config.sequence_length, config.input_size)).astype(
            np.float32
        ),
        rng.normal(size=(n_samples, config.forecast_horizon)).astype(np.float32),
        StandardScaler(),
        config,
    )
    assert result.stopped_epoch == 2
    assert 1 <= result.best_epoch <= result.stopped_epoch
    assert result.stopped_epoch <= config.epochs


def test_patchtst_tiny_train_records_epoch_counters() -> None:
    from dataclasses import replace

    result = fit_weekly_patchtst(
        synthetic_weekly_data(), replace(INDIA_WEEKLY_CONFIG, epochs=2, batch_size=128)
    )
    assert result.stopped_epoch == 2
    assert 1 <= result.best_epoch <= result.stopped_epoch
    assert np.isfinite(result.val_rank_ic)


def test_lstm_patience_one_can_stop_before_max_epochs() -> None:
    rng = np.random.default_rng(1)
    n_samples = 24
    config = LSTMConfig(
        epochs=8,
        early_stopping_patience=1,
        batch_size=8,
        hidden_size=8,
    )
    result = train_lstm(
        rng.normal(size=(n_samples, config.sequence_length, config.input_size)).astype(
            np.float32
        ),
        rng.normal(size=(n_samples, config.forecast_horizon)).astype(np.float32),
        StandardScaler(),
        config,
    )
    assert result.stopped_epoch <= config.epochs
    assert result.best_epoch <= result.stopped_epoch
    assert result.best_epoch >= 1


def test_create_training_metadata_hash_ignores_epoch_counters() -> None:
    """Version identity hashes config, not metrics: epoch fields must not drift IDs."""
    kwargs = {
        "model_type": "patchtst_halal_new",
        "version": "v2026-08-15-bbbbbbbbbbbb",
        "data_window_start": "2016-01-01",
        "data_window_end": "2025-12-31",
        "symbols": ["AAPL", "MSFT"],
        "config_dict": {"epochs": 100, "learning_rate": 0.0003},
        "train_loss": 0.01,
        "val_loss": 0.02,
        "baseline_loss": 0.03,
        "promoted": True,
        "prior_version": None,
    }
    a = create_training_metadata(**kwargs, best_epoch=2, stopped_epoch=17)
    b = create_training_metadata(
        **kwargs, best_epoch=58, stopped_epoch=73, val_rank_ic=0.41
    )
    assert a["config_symbols_hash"] == b["config_symbols_hash"]
    assert date.fromisoformat(a["data_window"]["start"]) == date(2016, 1, 1)
    assert "val_rank_ic" not in a["metrics"]
    assert b["metrics"]["val_rank_ic"] == 0.41

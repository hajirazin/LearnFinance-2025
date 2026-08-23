"""Epoch counters on forecaster TrainingResult and metadata.json."""

from datetime import date, timedelta

import numpy as np
from sklearn.preprocessing import StandardScaler

from brain_api.core.lstm.config import LSTMConfig
from brain_api.core.lstm.training import train_model_pytorch as train_lstm
from brain_api.core.patchtst.config import PatchTSTConfig
from brain_api.core.patchtst.training import train_model_pytorch as train_patchtst
from brain_api.storage.metadata import create_training_metadata


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
    rng = np.random.default_rng(0)
    config = PatchTSTConfig(
        context_length=32,
        patch_length=8,
        stride=8,
        d_model=16,
        num_attention_heads=2,
        num_hidden_layers=1,
        ffn_dim=32,
        epochs=2,
        early_stopping_patience=10,
        batch_size=4,
    )
    n_channels = config.num_input_channels
    symbols_list = ("AAA", "BBB", "CCC", "DDD")
    train_weeks = 4
    n_samples = train_weeks * len(symbols_list) + len(symbols_list)
    X = rng.normal(size=(n_samples, config.context_length, n_channels)).astype(
        np.float32
    )
    y = rng.normal(size=(n_samples, config.prediction_length, n_channels)).astype(
        np.float32
    )
    dates: list[date] = []
    labels: list[str] = []
    for week in range(train_weeks):
        week_date = date(2024, 1, 5) + timedelta(days=7 * week)
        for symbol in symbols_list:
            dates.append(week_date)
            labels.append(symbol)
    val_date = date(2024, 3, 1)
    for symbol in symbols_list:
        dates.append(val_date)
        labels.append(symbol)
    result = train_patchtst(
        X,
        y,
        StandardScaler(),
        config,
        anchor_dates=np.array(dates, dtype=object),
        sample_symbols=np.array(labels, dtype=object),
    )
    assert result.stopped_epoch == 2
    assert 1 <= result.best_epoch <= result.stopped_epoch
    assert result.stopped_epoch <= config.epochs
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

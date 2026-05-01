"""API-level tests for India PatchTST training endpoint."""

import os
import tempfile
from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.preprocessing import StandardScaler
from transformers import PatchTSTForPrediction

from brain_api.core.model_buckets import ModelType, get_bucket
from brain_api.core.patchtst import DatasetResult, TrainingResult
from brain_api.main import app
from brain_api.routes.training.dependencies import (
    get_patchtst_dataset_builder,
    get_patchtst_price_loader,
    get_patchtst_trainer,
)
from brain_api.storage.patchtst.local import PatchTSTNiftyShariah500ModelStorage


def _mock_india_symbols() -> list[str]:
    return ["INFY.NS", "TCS.NS"]


def _mock_india_symbols_missing_suffix() -> list[str]:
    """Bad universe resolver -- returns non-NSE symbols.

    Triggers the per-bucket symbol validator that raises 422 to prevent
    silently fetching the wrong instruments from yfinance (per AGENTS.md
    rule #1: no silent fallbacks).
    """
    return ["INFY", "TCS"]


def _mock_price_loader(symbols, start_date, end_date):
    import pandas as pd

    dates = pd.date_range(start=start_date, end=end_date, freq="B")[:100]
    prices = {}
    for symbol in symbols:
        prices[symbol] = pd.DataFrame(
            {
                "open": [100.0] * len(dates),
                "high": [101.0] * len(dates),
                "low": [99.0] * len(dates),
                "close": [100.5] * len(dates),
                "volume": [1000000] * len(dates),
            },
            index=dates,
        )
    return prices


def _mock_dataset_builder(aligned_features, prices, config) -> DatasetResult:
    n_samples = 10
    return DatasetResult(
        X=np.random.randn(n_samples, config.context_length, 5),
        y=np.random.randn(n_samples, 5, 5),
        feature_scaler=StandardScaler(),
    )


def _mock_trainer(X, y, feature_scaler, config, shutdown_event=None) -> TrainingResult:
    hf_config = config.to_hf_config()
    model = PatchTSTForPrediction(hf_config)
    return TrainingResult(
        model=model,
        feature_scaler=feature_scaler if feature_scaler else StandardScaler(),
        config=config,
        train_loss=0.01,
        val_loss=0.02,
        baseline_loss=0.05,
    )


def _mock_trainer_worse(
    X, y, feature_scaler, config, shutdown_event=None
) -> TrainingResult:
    hf_config = config.to_hf_config()
    model = PatchTSTForPrediction(hf_config)
    return TrainingResult(
        model=model,
        feature_scaler=feature_scaler if feature_scaler else StandardScaler(),
        config=config,
        train_loss=0.10,
        val_loss=0.10,
        baseline_loss=0.05,
    )


@pytest.fixture
def temp_india_storage():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield PatchTSTNiftyShariah500ModelStorage(base_path=tmpdir)


def _override_india_patchtst_bucket(
    monkeypatch, temp_storage, symbols_fn=_mock_india_symbols
):
    """Swap the ``(PATCHTST, nifty_shariah_500)`` registry entry.

    Keeps the production validator (``_validate_ns_suffix``) wired so
    suffix-enforcement tests still exercise that code path.
    """
    from brain_api.core import model_buckets

    original = get_bucket(ModelType.PATCHTST, "nifty_shariah_500")
    patched = replace(
        original,
        local_storage_class=lambda: temp_storage,
        symbols_resolver=symbols_fn,
    )
    monkeypatch.setitem(
        model_buckets._BUCKETS,
        (ModelType.PATCHTST, "nifty_shariah_500"),
        patched,
    )


@pytest.fixture
def client_india(temp_india_storage, monkeypatch):
    _override_india_patchtst_bucket(monkeypatch, temp_india_storage)
    app.dependency_overrides[get_patchtst_price_loader] = lambda: _mock_price_loader
    app.dependency_overrides[get_patchtst_dataset_builder] = (
        lambda: _mock_dataset_builder
    )
    app.dependency_overrides[get_patchtst_trainer] = lambda: _mock_trainer

    os.environ["LSTM_TRAIN_LOOKBACK_YEARS"] = "10"
    os.environ["LSTM_TRAIN_WINDOW_END_DATE"] = "2025-01-01"

    client = TestClient(app)
    yield client

    app.dependency_overrides.clear()
    os.environ.pop("LSTM_TRAIN_LOOKBACK_YEARS", None)
    os.environ.pop("LSTM_TRAIN_WINDOW_END_DATE", None)


TRAIN_INDIA_URL = "/train/patchtst/india?skip_snapshot=true"


def test_train_patchtst_india_returns_202(client_india):
    """POST /train/patchtst/india returns 202 (training runs in background)."""
    response = client_india.post(TRAIN_INDIA_URL)
    assert response.status_code == 202


def test_train_patchtst_india_explicit_universe_returns_202(client_india):
    """POST /train/patchtst/india with explicit registered universe returns 202."""
    response = client_india.post(
        TRAIN_INDIA_URL, json={"universe": "nifty_shariah_500"}
    )
    assert response.status_code == 202


def test_train_patchtst_india_unknown_universe_returns_422(client_india):
    """Unknown universe in request body must be rejected with 422."""
    response = client_india.post(TRAIN_INDIA_URL, json={"universe": "halal_new"})
    assert response.status_code == 422
    assert "halal_new" in response.text


def test_train_patchtst_india_rejects_missing_ns_suffix(
    temp_india_storage, monkeypatch
):
    """Symbol resolver returning non-NSE tickers must trip the bucket
    validator and surface as 422 (no silent fallback).
    """
    _override_india_patchtst_bucket(
        monkeypatch, temp_india_storage, symbols_fn=_mock_india_symbols_missing_suffix
    )
    app.dependency_overrides[get_patchtst_price_loader] = lambda: _mock_price_loader
    app.dependency_overrides[get_patchtst_dataset_builder] = (
        lambda: _mock_dataset_builder
    )
    app.dependency_overrides[get_patchtst_trainer] = lambda: _mock_trainer

    os.environ["LSTM_TRAIN_LOOKBACK_YEARS"] = "10"
    os.environ["LSTM_TRAIN_WINDOW_END_DATE"] = "2025-01-01"

    client = TestClient(app)
    try:
        response = client.post(TRAIN_INDIA_URL)
        assert response.status_code == 422
        assert ".NS" in response.text
    finally:
        app.dependency_overrides.clear()
        os.environ.pop("LSTM_TRAIN_LOOKBACK_YEARS", None)
        os.environ.pop("LSTM_TRAIN_WINDOW_END_DATE", None)


def test_train_patchtst_india_returns_required_fields(client_india):
    """POST /train/patchtst/india returns all required PatchTSTTrainResponse fields."""
    response1 = client_india.post(TRAIN_INDIA_URL)
    assert response1.status_code == 202
    response = client_india.post(TRAIN_INDIA_URL)
    assert response.status_code == 200

    data = response.json()
    assert "version" in data
    assert "data_window_start" in data
    assert "data_window_end" in data
    assert "metrics" in data
    assert "promoted" in data
    assert "num_input_channels" in data
    assert "signals_used" in data
    assert data["num_input_channels"] == 5
    assert data["signals_used"] == ["ohlcv"]


def test_train_patchtst_india_idempotent_version(client_india):
    """Calling POST /train/patchtst/india twice returns the same version."""
    r1 = client_india.post(TRAIN_INDIA_URL)
    assert r1.status_code == 202
    r2 = client_india.post(TRAIN_INDIA_URL)
    assert r2.status_code == 200
    r3 = client_india.post(TRAIN_INDIA_URL)
    assert r3.status_code == 200
    assert r2.json()["version"] == r3.json()["version"]


def test_train_patchtst_india_first_model_always_promoted(client_india):
    """First India PatchTST model is always promoted."""
    response1 = client_india.post(TRAIN_INDIA_URL)
    assert response1.status_code == 202
    response = client_india.post(TRAIN_INDIA_URL)
    assert response.status_code == 200
    assert response.json()["promoted"] is True


def test_train_patchtst_india_not_promoted_when_worse(monkeypatch):
    """India PatchTST model is NOT promoted when worse than prior."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PatchTSTNiftyShariah500ModelStorage(base_path=tmpdir)

        app.dependency_overrides.clear()
        _override_india_patchtst_bucket(monkeypatch, storage)
        app.dependency_overrides[get_patchtst_price_loader] = lambda: _mock_price_loader
        app.dependency_overrides[get_patchtst_dataset_builder] = (
            lambda: _mock_dataset_builder
        )
        app.dependency_overrides[get_patchtst_trainer] = lambda: _mock_trainer

        os.environ["LSTM_TRAIN_LOOKBACK_YEARS"] = "10"
        os.environ["LSTM_TRAIN_WINDOW_END_DATE"] = "2025-06-15"

        client = TestClient(app)
        train_url = "/train/patchtst/india?skip_snapshot=true"

        try:
            r1 = client.post(train_url)
            assert r1.status_code == 202
            r1b = client.post(train_url)
            assert r1b.status_code == 200
            first_version = r1b.json()["version"]
            assert storage.read_current_version() == first_version

            app.dependency_overrides[get_patchtst_trainer] = lambda: _mock_trainer_worse
            os.environ["LSTM_TRAIN_WINDOW_END_DATE"] = "2025-06-23"

            r2 = client.post(train_url)
            assert r2.status_code == 202
            r2b = client.post(train_url)
            assert r2b.status_code == 200
            assert r2b.json()["promoted"] is False
            assert storage.read_current_version() == first_version
        finally:
            app.dependency_overrides.clear()
            os.environ.pop("LSTM_TRAIN_LOOKBACK_YEARS", None)
            os.environ.pop("LSTM_TRAIN_WINDOW_END_DATE", None)


def test_train_patchtst_india_uses_india_storage(client_india, temp_india_storage):
    """India PatchTST uses the patchtst_nifty_shariah_500 storage directory."""
    response1 = client_india.post(TRAIN_INDIA_URL)
    assert response1.status_code == 202
    response = client_india.post(TRAIN_INDIA_URL)
    assert response.status_code == 200

    version = response.json()["version"]
    assert temp_india_storage.version_exists(version)
    assert temp_india_storage.model_type == "patchtst_nifty_shariah_500"


def test_train_patchtst_india_version_differs_from_us():
    """India PatchTST version differs from US PatchTST (different symbols)."""
    from datetime import date

    from brain_api.core.patchtst import DEFAULT_CONFIG
    from brain_api.core.patchtst import compute_version as patchtst_compute_version

    start = date(2015, 1, 1)
    end = date(2025, 1, 1)

    us_version = patchtst_compute_version(start, end, ["AAPL", "MSFT"], DEFAULT_CONFIG)
    india_version = patchtst_compute_version(
        start, end, ["INFY.NS", "TCS.NS"], DEFAULT_CONFIG
    )

    assert us_version != india_version

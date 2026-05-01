"""API-level tests for PatchTST training endpoint."""

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
from brain_api.routes.training import (
    get_patchtst_dataset_builder,
    get_patchtst_price_loader,
    get_patchtst_trainer,
)
from brain_api.storage.local import PatchTSTHalalNewModelStorage

# ============================================================================
# Test fixtures and mocks
# ============================================================================


def mock_symbols() -> list[str]:
    """Return a small fixed list of symbols for testing."""
    return ["AAPL", "MSFT"]


def mock_price_loader(symbols, start_date, end_date):
    """Return mock price data for testing."""
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


def mock_dataset_builder(aligned_features, prices, config) -> DatasetResult:
    """Return a mock dataset result for 5-channel OHLCV multi-task prediction."""
    n_samples = 10
    return DatasetResult(
        X=np.random.randn(n_samples, config.context_length, 5),
        y=np.random.randn(n_samples, 5, 5),
        feature_scaler=StandardScaler(),
    )


def mock_trainer(X, y, feature_scaler, config, shutdown_event=None) -> TrainingResult:
    """Return a mock training result with controllable metrics."""
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


def mock_trainer_worse_than_baseline(
    X, y, feature_scaler, config, shutdown_event=None
) -> TrainingResult:
    """Return a mock training result that is worse than baseline."""
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
def temp_storage():
    """Create a temporary storage directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield PatchTSTHalalNewModelStorage(base_path=tmpdir)


def _override_us_patchtst_bucket(monkeypatch, temp_storage, symbols_fn=mock_symbols):
    """Swap the ``(PATCHTST, halal_new)`` registry entry for tests.

    Symbol resolution and storage are now driven by the bucket registry
    rather than ``Depends`` overrides; tests therefore monkey-patch the
    bucket entry itself so the production registry stays untouched
    between tests.
    """
    from brain_api.core import model_buckets

    original = get_bucket(ModelType.PATCHTST, "halal_new")
    patched = replace(
        original,
        local_storage_class=lambda: temp_storage,
        symbols_resolver=symbols_fn,
    )
    monkeypatch.setitem(
        model_buckets._BUCKETS,
        (ModelType.PATCHTST, "halal_new"),
        patched,
    )


@pytest.fixture
def client_with_mocks(temp_storage, monkeypatch):
    """Create test client with mocked dependencies."""
    _override_us_patchtst_bucket(monkeypatch, temp_storage)
    app.dependency_overrides[get_patchtst_price_loader] = lambda: mock_price_loader
    app.dependency_overrides[get_patchtst_dataset_builder] = (
        lambda: mock_dataset_builder
    )
    app.dependency_overrides[get_patchtst_trainer] = lambda: mock_trainer

    os.environ["LSTM_TRAIN_LOOKBACK_YEARS"] = "10"
    os.environ["LSTM_TRAIN_WINDOW_END_DATE"] = "2025-01-01"

    client = TestClient(app)
    yield client

    app.dependency_overrides.clear()
    os.environ.pop("LSTM_TRAIN_LOOKBACK_YEARS", None)
    os.environ.pop("LSTM_TRAIN_WINDOW_END_DATE", None)


# ============================================================================
# Scenario 1: Empty-body success + resolved window
# ============================================================================


TRAIN_URL = "/train/patchtst?skip_snapshot=true"


def test_train_patchtst_empty_body_returns_202(client_with_mocks):
    """POST /train/patchtst with empty body returns 202 (training runs in background)."""
    response = client_with_mocks.post(TRAIN_URL, json={})
    assert response.status_code == 202


def test_train_patchtst_no_body_returns_202(client_with_mocks):
    """POST /train/patchtst with no body returns 202 (training runs in background)."""
    response = client_with_mocks.post(TRAIN_URL)
    assert response.status_code == 202


def test_train_patchtst_explicit_universe_returns_202(client_with_mocks):
    """POST /train/patchtst with the registered US universe returns 202."""
    response = client_with_mocks.post(TRAIN_URL, json={"universe": "halal_new"})
    assert response.status_code == 202


def test_train_patchtst_unknown_universe_returns_422(client_with_mocks):
    """Unknown universe in the request body must be rejected with 422."""
    response = client_with_mocks.post(TRAIN_URL, json={"universe": "halal_filtered"})
    assert response.status_code == 422
    assert "halal_filtered" in response.text


def test_train_patchtst_us_endpoint_rejects_india_universe(client_with_mocks):
    """The US endpoint must reject the India universe even though it is
    registered for the PatchTST family. Each market gets its own route.
    """
    response = client_with_mocks.post(TRAIN_URL, json={"universe": "nifty_shariah_500"})
    assert response.status_code == 422
    assert "nifty_shariah_500" in response.text


def test_train_patchtst_returns_resolved_window(client_with_mocks):
    """POST /train/patchtst returns Friday-anchored data_window_end from config."""
    response1 = client_with_mocks.post(TRAIN_URL, json={})
    assert response1.status_code == 202
    response = client_with_mocks.post(TRAIN_URL, json={})
    assert response.status_code == 200

    data = response.json()
    assert "data_window_start" in data
    assert "data_window_end" in data

    assert data["data_window_end"] == "2024-12-27"
    assert data["data_window_start"] == "2014-01-01"


def test_train_patchtst_returns_required_fields(client_with_mocks):
    """POST /train/patchtst returns all required response fields."""
    response1 = client_with_mocks.post(TRAIN_URL, json={})
    assert response1.status_code == 202
    response = client_with_mocks.post(TRAIN_URL, json={})
    assert response.status_code == 200

    data = response.json()
    assert "version" in data
    assert "data_window_start" in data
    assert "data_window_end" in data
    assert "metrics" in data
    assert "promoted" in data
    assert "num_input_channels" in data
    assert "signals_used" in data

    assert isinstance(data["metrics"], dict)

    assert data["num_input_channels"] == 5
    assert data["signals_used"] == ["ohlcv"]


# ============================================================================
# Scenario 2: Idempotency on rerun
# ============================================================================


def test_train_patchtst_idempotent_version(client_with_mocks):
    """Calling POST /train/patchtst twice returns the same version."""
    response1 = client_with_mocks.post(TRAIN_URL, json={})
    assert response1.status_code == 202

    response2 = client_with_mocks.post(TRAIN_URL, json={})
    assert response2.status_code == 200
    version2 = response2.json()["version"]

    response3 = client_with_mocks.post(TRAIN_URL, json={})
    assert response3.status_code == 200
    version3 = response3.json()["version"]

    assert version2 == version3, "Version should be identical on rerun with same config"


def test_train_patchtst_idempotent_does_not_change_current(
    client_with_mocks, temp_storage
):
    """Rerunning training does not change 'current' pointer if already promoted."""
    response1 = client_with_mocks.post(TRAIN_URL, json={})
    assert response1.status_code == 202

    response2 = client_with_mocks.post(TRAIN_URL, json={})
    assert response2.status_code == 200
    version2 = response2.json()["version"]

    current_after_first = temp_storage.read_current_version()

    response3 = client_with_mocks.post(TRAIN_URL, json={})
    assert response3.status_code == 200
    version3 = response3.json()["version"]

    current_after_second = temp_storage.read_current_version()

    assert version2 == version3
    assert current_after_first == current_after_second


# ============================================================================
# Scenario 3: Promotion gate behavior
# ============================================================================


def test_train_patchtst_first_model_always_promoted(client_with_mocks):
    """First model is always promoted (no prior model to compare against)."""
    response1 = client_with_mocks.post(TRAIN_URL, json={})
    assert response1.status_code == 202
    response = client_with_mocks.post(TRAIN_URL, json={})
    assert response.status_code == 200

    data = response.json()
    assert data["promoted"] is True


def test_train_patchtst_not_promoted_when_worse_than_prior(monkeypatch):
    """Model is NOT promoted when worse than prior model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fresh_storage = PatchTSTHalalNewModelStorage(base_path=tmpdir)

        app.dependency_overrides.clear()

        _override_us_patchtst_bucket(monkeypatch, fresh_storage)
        app.dependency_overrides[get_patchtst_price_loader] = lambda: mock_price_loader
        app.dependency_overrides[get_patchtst_dataset_builder] = (
            lambda: mock_dataset_builder
        )
        app.dependency_overrides[get_patchtst_trainer] = lambda: mock_trainer

        os.environ["LSTM_TRAIN_LOOKBACK_YEARS"] = "10"
        os.environ["LSTM_TRAIN_WINDOW_END_DATE"] = "2025-06-15"

        client = TestClient(app)

        try:
            train_url = "/train/patchtst?skip_snapshot=true"
            response1 = client.post(train_url, json={})
            assert response1.status_code == 202
            response1b = client.post(train_url, json={})
            assert response1b.status_code == 200
            first_version = response1b.json()["version"]
            assert fresh_storage.read_current_version() == first_version

            app.dependency_overrides[get_patchtst_trainer] = (
                lambda: mock_trainer_worse_than_baseline
            )
            os.environ["LSTM_TRAIN_WINDOW_END_DATE"] = "2025-06-23"

            response2 = client.post(train_url, json={})
            assert response2.status_code == 202
            response2b = client.post(train_url, json={})
            assert response2b.status_code == 200

            data = response2b.json()
            assert data["promoted"] is False

            current = fresh_storage.read_current_version()
            assert current == first_version
        finally:
            app.dependency_overrides.clear()
            os.environ.pop("LSTM_TRAIN_LOOKBACK_YEARS", None)
            os.environ.pop("LSTM_TRAIN_WINDOW_END_DATE", None)


def test_train_patchtst_current_unchanged_when_not_promoted(temp_storage, monkeypatch):
    """The 'current' pointer is unchanged when promotion fails."""
    app.dependency_overrides.clear()

    _override_us_patchtst_bucket(monkeypatch, temp_storage)
    app.dependency_overrides[get_patchtst_price_loader] = lambda: mock_price_loader
    app.dependency_overrides[get_patchtst_dataset_builder] = (
        lambda: mock_dataset_builder
    )
    app.dependency_overrides[get_patchtst_trainer] = lambda: mock_trainer

    os.environ["LSTM_TRAIN_LOOKBACK_YEARS"] = "10"
    os.environ["LSTM_TRAIN_WINDOW_END_DATE"] = "2025-01-01"

    client = TestClient(app)

    train_url = "/train/patchtst?skip_snapshot=true"
    response1 = client.post(train_url, json={})
    assert response1.status_code == 202
    response1b = client.post(train_url, json={})
    assert response1b.status_code == 200
    promoted_version = response1b.json()["version"]

    current_before = temp_storage.read_current_version()
    assert current_before == promoted_version

    app.dependency_overrides[get_patchtst_trainer] = (
        lambda: mock_trainer_worse_than_baseline
    )
    os.environ["LSTM_TRAIN_WINDOW_END_DATE"] = "2025-01-13"

    response2 = client.post(train_url, json={})
    assert response2.status_code == 202
    response2b = client.post(train_url, json={})
    assert response2b.status_code == 200
    data2 = response2b.json()

    assert data2["promoted"] is False

    current_after = temp_storage.read_current_version()
    assert current_after == promoted_version

    app.dependency_overrides.clear()
    os.environ.pop("LSTM_TRAIN_LOOKBACK_YEARS", None)
    os.environ.pop("LSTM_TRAIN_WINDOW_END_DATE", None)


# ============================================================================
# Scenario 4: PatchTST-specific behavior
# ============================================================================


def test_train_patchtst_version_differs_from_lstm():
    """PatchTST version hash differs from LSTM even with same window/symbols."""
    from datetime import date

    from brain_api.core.lstm import DEFAULT_CONFIG as LSTM_DEFAULT_CONFIG
    from brain_api.core.lstm import compute_version as lstm_compute_version
    from brain_api.core.patchtst import DEFAULT_CONFIG as PATCHTST_DEFAULT_CONFIG
    from brain_api.core.patchtst import compute_version as patchtst_compute_version

    start = date(2015, 1, 1)
    end = date(2025, 1, 1)
    symbols = ["AAPL", "MSFT"]

    lstm_version = lstm_compute_version(start, end, symbols, LSTM_DEFAULT_CONFIG)
    patchtst_version = patchtst_compute_version(
        start, end, symbols, PATCHTST_DEFAULT_CONFIG
    )

    assert lstm_version != patchtst_version, "PatchTST and LSTM versions should differ"

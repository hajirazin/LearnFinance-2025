"""API-level tests for LSTM training endpoint."""

import os
import tempfile

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.preprocessing import StandardScaler

from brain_api.core.lstm import DatasetResult, LSTMModel, TrainingResult
from brain_api.main import app
from brain_api.routes.training import (
    get_dataset_builder,
    get_lstm_training_symbols,
    get_price_loader,
    get_storage,
    get_trainer,
)
from brain_api.storage.local import LocalModelStorage

# ============================================================================
# Test fixtures and mocks
# ============================================================================


def mock_symbols() -> list[str]:
    """Return a small fixed list of symbols for testing."""
    return ["AAPL", "MSFT"]


def mock_price_loader(symbols, start_date, end_date):
    """Return mock price data for testing."""
    import pandas as pd

    # Create minimal fake price data for each symbol
    dates = pd.date_range(start=start_date, end=end_date, freq="B")[
        :100
    ]  # 100 business days
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


def mock_dataset_builder(prices, config) -> DatasetResult:
    """Return a mock dataset result for direct 5-day close-return prediction."""
    # Return non-empty arrays to pass validation checks
    n_samples = 10
    return DatasetResult(
        X=np.random.randn(n_samples, config.sequence_length, config.input_size),
        y=np.random.randn(n_samples, 5),  # 5 close log returns per sample
        feature_scaler=StandardScaler(),
    )


def mock_trainer(X, y, feature_scaler, config) -> TrainingResult:
    """Return a mock training result with controllable metrics."""
    model = LSTMModel(config)
    return TrainingResult(
        model=model,
        feature_scaler=feature_scaler if feature_scaler else StandardScaler(),
        config=config,
        train_loss=0.01,
        val_loss=0.02,  # Better than baseline (0.05)
        baseline_loss=0.05,
    )


def mock_trainer_worse_than_baseline(X, y, feature_scaler, config) -> TrainingResult:
    """Return a mock training result that is worse than baseline."""
    model = LSTMModel(config)
    return TrainingResult(
        model=model,
        feature_scaler=feature_scaler if feature_scaler else StandardScaler(),
        config=config,
        train_loss=0.10,
        val_loss=0.10,  # Worse than baseline (0.05)
        baseline_loss=0.05,
    )


@pytest.fixture
def temp_storage():
    """Create a temporary storage directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield LocalModelStorage(base_path=tmpdir)


@pytest.fixture
def client_with_mocks(temp_storage):
    """Create test client with mocked dependencies."""
    # Override dependencies
    app.dependency_overrides[get_storage] = lambda: temp_storage
    app.dependency_overrides[get_lstm_training_symbols] = mock_symbols
    app.dependency_overrides[get_price_loader] = lambda: mock_price_loader
    app.dependency_overrides[get_dataset_builder] = lambda: mock_dataset_builder
    app.dependency_overrides[get_trainer] = lambda: mock_trainer

    # Set fixed window for deterministic tests
    os.environ["LSTM_TRAIN_LOOKBACK_YEARS"] = "10"
    os.environ["LSTM_TRAIN_WINDOW_END_DATE"] = "2025-01-01"

    client = TestClient(app)
    yield client

    # Cleanup
    app.dependency_overrides.clear()
    os.environ.pop("LSTM_TRAIN_LOOKBACK_YEARS", None)
    os.environ.pop("LSTM_TRAIN_WINDOW_END_DATE", None)


# ============================================================================
# Scenario 1: Empty-body success + resolved window
# ============================================================================


def test_train_lstm_empty_body_returns_200(client_with_mocks):
    """POST /train/lstm with empty body returns 200."""
    response = client_with_mocks.post("/train/lstm", json={})
    assert response.status_code == 200


def test_train_lstm_no_body_returns_200(client_with_mocks):
    """POST /train/lstm with no body returns 200."""
    response = client_with_mocks.post("/train/lstm")
    assert response.status_code == 200


def test_train_lstm_returns_resolved_window(client_with_mocks):
    """POST /train/lstm returns Friday-anchored data_window_end from config."""
    response = client_with_mocks.post("/train/lstm", json={})
    assert response.status_code == 200

    data = response.json()
    assert "data_window_start" in data
    assert "data_window_end" in data

    # Verify window is Friday-anchored.
    # Env config sets end_date to 2025-01-01 (Wednesday), which anchors to 2024-12-27 (Friday)
    assert data["data_window_end"] == "2024-12-27"
    assert data["data_window_start"] == "2014-01-01"  # 10 years before 2024


def test_train_lstm_returns_required_fields(client_with_mocks):
    """POST /train/lstm returns all required response fields."""
    response = client_with_mocks.post("/train/lstm", json={})
    assert response.status_code == 200

    data = response.json()
    assert "version" in data
    assert "data_window_start" in data
    assert "data_window_end" in data
    assert "metrics" in data
    assert "promoted" in data
    # prior_version can be None

    # Check metrics structure
    assert isinstance(data["metrics"], dict)


# ============================================================================
# Scenario 2: Idempotency on rerun
# ============================================================================


def test_train_lstm_idempotent_version(client_with_mocks):
    """Calling POST /train/lstm twice returns the same version."""
    response1 = client_with_mocks.post("/train/lstm", json={})
    assert response1.status_code == 200
    version1 = response1.json()["version"]

    response2 = client_with_mocks.post("/train/lstm", json={})
    assert response2.status_code == 200
    version2 = response2.json()["version"]

    assert version1 == version2, "Version should be identical on rerun with same config"


def test_train_lstm_idempotent_does_not_change_current(client_with_mocks, temp_storage):
    """Rerunning training does not change 'current' pointer if already promoted."""
    # First call - should promote
    response1 = client_with_mocks.post("/train/lstm", json={})
    assert response1.status_code == 200
    data1 = response1.json()
    version1 = data1["version"]

    current_after_first = temp_storage.read_current_version()

    # Second call - should return same version without changing current
    response2 = client_with_mocks.post("/train/lstm", json={})
    assert response2.status_code == 200
    version2 = response2.json()["version"]

    current_after_second = temp_storage.read_current_version()

    assert version1 == version2
    assert current_after_first == current_after_second


# ============================================================================
# Scenario 3: Promotion gate behavior
# ============================================================================


def test_train_lstm_first_model_always_promoted(client_with_mocks):
    """First model is always promoted (no prior model to compare against)."""
    response = client_with_mocks.post("/train/lstm", json={})
    assert response.status_code == 200

    data = response.json()
    assert data["promoted"] is True


def test_train_lstm_not_promoted_when_worse_than_prior():
    """Model is NOT promoted when worse than prior model."""
    # Use a fresh temp storage to avoid conflicts with other tests
    with tempfile.TemporaryDirectory() as tmpdir:
        fresh_storage = LocalModelStorage(base_path=tmpdir)

        # Clear any leftover overrides from previous tests
        app.dependency_overrides.clear()

        # First, create a good model that gets promoted (first model always promoted)
        app.dependency_overrides[get_storage] = lambda: fresh_storage
        app.dependency_overrides[get_lstm_training_symbols] = mock_symbols
        app.dependency_overrides[get_price_loader] = lambda: mock_price_loader
        app.dependency_overrides[get_dataset_builder] = lambda: mock_dataset_builder
        app.dependency_overrides[get_trainer] = lambda: mock_trainer

        os.environ["LSTM_TRAIN_LOOKBACK_YEARS"] = "10"
        os.environ["LSTM_TRAIN_WINDOW_END_DATE"] = (
            "2025-06-15"  # Sunday -> June 13 (Fri)
        )

        client = TestClient(app)

        try:
            # Train first model - should be promoted (first model always promoted)
            response1 = client.post("/train/lstm", json={})
            assert response1.status_code == 200
            first_version = response1.json()["version"]
            assert fresh_storage.read_current_version() == first_version

            # Now train a worse model with different date (to generate new version)
            # June 23, 2025 is Monday -> anchors to June 20 (Fri), different from June 13
            # mock_trainer_worse_than_baseline returns val_loss=0.10, which is > 0.02 (prior)
            app.dependency_overrides[get_trainer] = (
                lambda: mock_trainer_worse_than_baseline
            )
            os.environ["LSTM_TRAIN_WINDOW_END_DATE"] = "2025-06-23"

            response2 = client.post("/train/lstm", json={})
            assert response2.status_code == 200

            data = response2.json()
            # Mock trainer returns val_loss=0.10 > prior=0.02
            assert data["promoted"] is False

            # Current should still point to the first version (worse model not promoted)
            current = fresh_storage.read_current_version()
            assert current == first_version
        finally:
            app.dependency_overrides.clear()
            os.environ.pop("LSTM_TRAIN_LOOKBACK_YEARS", None)
            os.environ.pop("LSTM_TRAIN_WINDOW_END_DATE", None)


def test_train_lstm_current_unchanged_when_not_promoted(temp_storage):
    """The 'current' pointer is unchanged when promotion fails."""
    # Clear any leftover overrides from previous tests
    app.dependency_overrides.clear()

    # First, create a good model that gets promoted
    app.dependency_overrides[get_storage] = lambda: temp_storage
    app.dependency_overrides[get_lstm_training_symbols] = mock_symbols
    app.dependency_overrides[get_price_loader] = lambda: mock_price_loader
    app.dependency_overrides[get_dataset_builder] = lambda: mock_dataset_builder
    app.dependency_overrides[get_trainer] = lambda: mock_trainer

    os.environ["LSTM_TRAIN_LOOKBACK_YEARS"] = "10"
    os.environ["LSTM_TRAIN_WINDOW_END_DATE"] = "2025-01-01"

    client = TestClient(app)

    response1 = client.post("/train/lstm", json={})
    assert response1.status_code == 200
    promoted_version = response1.json()["version"]

    # Verify it was promoted
    current_before = temp_storage.read_current_version()
    assert current_before == promoted_version

    # Now try with a different window that produces worse results
    # Use a date that anchors to a DIFFERENT Friday (Jan 10, 2025 = Friday -> Dec 27, 2024)
    # but Jan 13, 2025 = Monday -> Jan 10, 2025 (Friday) which is different from Dec 27
    app.dependency_overrides[get_trainer] = lambda: mock_trainer_worse_than_baseline
    os.environ["LSTM_TRAIN_WINDOW_END_DATE"] = (
        "2025-01-13"  # Monday -> anchors to Jan 10 (Friday), different from Dec 27
    )

    response2 = client.post("/train/lstm", json={})
    assert response2.status_code == 200
    data2 = response2.json()

    # Should not be promoted (worse than prior)
    assert data2["promoted"] is False

    # Current should still point to the first version
    current_after = temp_storage.read_current_version()
    assert current_after == promoted_version

    # Cleanup
    app.dependency_overrides.clear()
    os.environ.pop("LSTM_TRAIN_LOOKBACK_YEARS", None)
    os.environ.pop("LSTM_TRAIN_WINDOW_END_DATE", None)

"""API-level tests for PatchTST inference endpoint."""

import tempfile

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.preprocessing import StandardScaler
from transformers import PatchTSTForPrediction

from brain_api.core.patchtst import DEFAULT_CONFIG, PatchTSTConfig
from brain_api.main import app
from brain_api.routes.inference import get_patchtst_storage
from brain_api.storage.local import PatchTSTModelStorage

# ============================================================================
# Test fixtures and mocks
# ============================================================================


def create_mock_patchtst_artifacts(
    storage: PatchTSTModelStorage, config: PatchTSTConfig
) -> str:
    """Create and write mock PatchTST model artifacts for testing.

    Returns the version string.
    """
    version = "v2025-01-01-patchtest"

    # Create a dummy model
    hf_config = config.to_hf_config()
    model = PatchTSTForPrediction(hf_config)

    # Create a fitted scaler (fit on dummy data)
    scaler = StandardScaler()
    dummy_data = np.random.randn(100, config.num_input_channels)
    scaler.fit(dummy_data)

    # Create metadata
    metadata = {
        "model_type": "patchtst",
        "version": version,
        "training_timestamp": "2025-01-01T00:00:00+00:00",
        "data_window": {"start": "2020-01-01", "end": "2025-01-01"},
        "symbols": ["AAPL", "MSFT"],
        "config": config.to_dict(),
        "metrics": {"train_loss": 0.01, "val_loss": 0.02, "baseline_loss": 0.05},
        "promoted": True,
        "prior_version": None,
    }

    # Write artifacts
    storage.write_artifacts(
        version=version,
        model=model,
        feature_scaler=scaler,
        config=config,
        metadata=metadata,
    )

    # Promote to current
    storage.promote_version(version)

    return version


def mock_price_loader(symbols, start_date, end_date):
    """Return mock price data for testing."""
    prices = {}

    date_range = pd.bdate_range(start=start_date, end=end_date)

    for symbol in symbols:
        if len(date_range) < 10:
            continue

        np.random.seed(hash(symbol) % 2**32)
        base_price = 100.0
        returns = np.random.randn(len(date_range)) * 0.02
        prices_array = base_price * np.exp(np.cumsum(returns))

        df = pd.DataFrame(
            {
                "open": prices_array * (1 + np.random.randn(len(date_range)) * 0.005),
                "high": prices_array
                * (1 + np.abs(np.random.randn(len(date_range)) * 0.01)),
                "low": prices_array
                * (1 - np.abs(np.random.randn(len(date_range)) * 0.01)),
                "close": prices_array,
                "volume": np.random.randint(1000000, 10000000, len(date_range)),
            },
            index=date_range,
        )
        prices[symbol] = df

    return prices


class MockSignalBuilder:
    """Mock RealTimeSignalBuilder for testing."""

    def build_news_dataframes(self, symbols, start_date, end_date):
        """Return mock news sentiment DataFrames for testing."""
        sentiment = {}
        for symbol in symbols:
            np.random.seed(hash(symbol + "news") % 2**32)
            df = pd.DataFrame(
                {"sentiment_score": [np.random.uniform(-0.5, 0.5)]},
                index=pd.DatetimeIndex([pd.Timestamp(end_date)]),
            )
            sentiment[symbol] = df
        return sentiment

    def build_fundamentals_dataframes(self, symbols, start_date, end_date):
        """Return mock fundamentals DataFrames for testing."""
        fundamentals = {}
        for symbol in symbols:
            np.random.seed(hash(symbol + "fund") % 2**32)
            df = pd.DataFrame(
                {
                    "gross_margin": [np.random.uniform(0.2, 0.6)],
                    "operating_margin": [np.random.uniform(0.1, 0.4)],
                    "net_margin": [np.random.uniform(0.05, 0.3)],
                    "current_ratio": [np.random.uniform(1.0, 3.0)],
                    "debt_to_equity": [np.random.uniform(0.1, 2.0)],
                },
                index=pd.DatetimeIndex([pd.Timestamp(end_date)]),
            )
            fundamentals[symbol] = df
        return fundamentals


@pytest.fixture
def temp_storage():
    """Create a temporary storage directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PatchTSTModelStorage(base_path=tmpdir)
        create_mock_patchtst_artifacts(storage, DEFAULT_CONFIG)
        yield storage


@pytest.fixture
def client_with_mocks(temp_storage, monkeypatch):
    """Create test client with mocked dependencies."""
    # Override storage dependency
    app.dependency_overrides[get_patchtst_storage] = lambda: temp_storage

    # Monkeypatch the data loaders in the inference module
    from brain_api.routes.inference import patchtst as inference_module

    monkeypatch.setattr(inference_module, "patchtst_load_prices", mock_price_loader)
    # Mock the RealTimeSignalBuilder class to return mock data
    monkeypatch.setattr(inference_module, "RealTimeSignalBuilder", MockSignalBuilder)

    client = TestClient(app)
    yield client

    # Cleanup
    app.dependency_overrides.clear()


# ============================================================================
# Scenario 1: Basic inference success
# ============================================================================


def test_inference_patchtst_returns_200(client_with_mocks):
    """POST /inference/patchtst with valid symbols returns 200."""
    response = client_with_mocks.post(
        "/inference/patchtst",
        json={"symbols": ["AAPL", "MSFT"]},
    )
    assert response.status_code == 200


def test_inference_patchtst_returns_predictions_for_all_symbols(client_with_mocks):
    """POST /inference/patchtst returns predictions for all requested symbols."""
    response = client_with_mocks.post(
        "/inference/patchtst",
        json={"symbols": ["AAPL", "MSFT", "GOOGL"]},
    )
    assert response.status_code == 200

    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 3

    symbols_returned = {p["symbol"] for p in data["predictions"]}
    assert symbols_returned == {"AAPL", "MSFT", "GOOGL"}


def test_inference_patchtst_returns_required_fields(client_with_mocks):
    """POST /inference/patchtst returns all required response fields."""
    response = client_with_mocks.post(
        "/inference/patchtst",
        json={"symbols": ["AAPL"]},
    )
    assert response.status_code == 200

    data = response.json()

    # Top-level fields
    assert "predictions" in data
    assert "model_version" in data
    assert "as_of_date" in data
    assert "target_week_start" in data
    assert "target_week_end" in data
    assert "signals_used" in data

    # Prediction fields
    assert len(data["predictions"]) == 1
    pred = data["predictions"][0]
    assert "symbol" in pred
    assert "predicted_weekly_return_pct" in pred
    assert "direction" in pred
    assert "has_enough_history" in pred
    assert "history_days_used" in pred
    assert "target_week_start" in pred
    assert "target_week_end" in pred
    assert "has_news_data" in pred
    assert "has_fundamentals_data" in pred


def test_inference_patchtst_returns_numeric_prediction(client_with_mocks):
    """POST /inference/patchtst returns numeric predicted weekly return percentage."""
    response = client_with_mocks.post(
        "/inference/patchtst",
        json={"symbols": ["AAPL"]},
    )
    assert response.status_code == 200

    data = response.json()
    pred = data["predictions"][0]

    assert pred["has_enough_history"] is True
    assert isinstance(pred["predicted_weekly_return_pct"], int | float)
    assert pred["direction"] in ["UP", "DOWN", "FLAT"]


def test_inference_patchtst_returns_signals_used(client_with_mocks):
    """POST /inference/patchtst returns list of signals that were available."""
    response = client_with_mocks.post(
        "/inference/patchtst",
        json={"symbols": ["AAPL"]},
    )
    assert response.status_code == 200

    data = response.json()
    assert "signals_used" in data
    assert "ohlcv" in data["signals_used"]
    # With mocks, news and fundamentals should also be present
    assert "news_sentiment" in data["signals_used"]
    assert "fundamentals" in data["signals_used"]


# ============================================================================
# Scenario 2: No current model
# ============================================================================


def test_inference_patchtst_no_model_returns_503():
    """POST /inference/patchtst returns 503 when no model is trained."""
    with tempfile.TemporaryDirectory() as tmpdir:
        empty_storage = PatchTSTModelStorage(base_path=tmpdir)

        app.dependency_overrides.clear()
        app.dependency_overrides[get_patchtst_storage] = lambda: empty_storage

        client = TestClient(app)

        try:
            response = client.post(
                "/inference/patchtst",
                json={"symbols": ["AAPL"]},
            )
            assert response.status_code == 503
            assert "No trained PatchTST model available" in response.json()["detail"]
        finally:
            app.dependency_overrides.clear()


# ============================================================================
# Scenario 3: Request validation
# ============================================================================


def test_inference_patchtst_empty_symbols_returns_422(client_with_mocks):
    """POST /inference/patchtst with empty symbols list returns 422."""
    response = client_with_mocks.post(
        "/inference/patchtst",
        json={"symbols": []},
    )
    assert response.status_code == 422


def test_inference_patchtst_no_symbols_returns_422(client_with_mocks):
    """POST /inference/patchtst without symbols field returns 422."""
    response = client_with_mocks.post(
        "/inference/patchtst",
        json={},
    )
    assert response.status_code == 422


def test_inference_patchtst_custom_as_of_date(client_with_mocks):
    """POST /inference/patchtst respects custom as_of_date."""
    response = client_with_mocks.post(
        "/inference/patchtst",
        json={"symbols": ["AAPL"], "as_of_date": "2025-01-06"},
    )
    assert response.status_code == 200

    data = response.json()
    assert data["as_of_date"] == "2025-01-06"


# ============================================================================
# Scenario 4: Sorting behavior
# ============================================================================


def test_inference_patchtst_predictions_sorted_by_return_desc(client_with_mocks):
    """POST /inference/patchtst returns predictions sorted by predicted_weekly_return_pct descending."""
    response = client_with_mocks.post(
        "/inference/patchtst",
        json={"symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]},
    )
    assert response.status_code == 200

    data = response.json()
    predictions = data["predictions"]

    # Extract returns for symbols with valid predictions
    valid_returns = [
        p["predicted_weekly_return_pct"]
        for p in predictions
        if p["predicted_weekly_return_pct"] is not None
    ]

    # Check that valid returns are sorted descending (highest first)
    assert valid_returns == sorted(valid_returns, reverse=True)


# ============================================================================
# Scenario 5: Signal availability flags
# ============================================================================


def test_inference_patchtst_signal_flags_in_predictions(client_with_mocks):
    """Each prediction includes flags for news and fundamentals data availability."""
    response = client_with_mocks.post(
        "/inference/patchtst",
        json={"symbols": ["AAPL"]},
    )
    assert response.status_code == 200

    data = response.json()
    pred = data["predictions"][0]

    # These flags should be present
    assert "has_news_data" in pred
    assert "has_fundamentals_data" in pred

    # With mocks, both should be True
    assert pred["has_news_data"] is True
    assert pred["has_fundamentals_data"] is True

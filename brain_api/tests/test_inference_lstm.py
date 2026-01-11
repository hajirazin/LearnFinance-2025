"""API-level tests for LSTM inference endpoint."""

import tempfile

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.preprocessing import StandardScaler

from brain_api.core.lstm import DEFAULT_CONFIG, LSTMConfig, LSTMModel
from brain_api.main import app
from brain_api.routes.inference import (
    get_price_loader,
    get_storage,
)
from brain_api.storage.local import LocalModelStorage

# ============================================================================
# Test fixtures and mocks
# ============================================================================


def create_mock_model_artifacts(storage: LocalModelStorage, config: LSTMConfig) -> str:
    """Create and write mock LSTM model artifacts for testing.

    Returns the version string.
    """
    version = "v2025-01-01-test123"

    # Create a dummy model
    model = LSTMModel(config)

    # Create a fitted scaler (fit on dummy data)
    scaler = StandardScaler()
    dummy_data = np.random.randn(100, config.input_size)
    scaler.fit(dummy_data)

    # Create metadata
    metadata = {
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

    # Generate enough mock data for inference (need 60+ trading days)
    # Create 100 trading days of data
    date_range = pd.bdate_range(start=start_date, end=end_date)

    for symbol in symbols:
        if len(date_range) < 10:
            continue

        # Generate random walk prices
        np.random.seed(hash(symbol) % 2**32)  # Deterministic per symbol
        base_price = 100.0
        returns = np.random.randn(len(date_range)) * 0.02  # 2% daily volatility
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


def mock_price_loader_no_data(symbols, start_date, end_date):
    """Return empty prices dict to simulate missing data."""
    return {}


@pytest.fixture
def temp_storage():
    """Create a temporary storage directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalModelStorage(base_path=tmpdir)
        # Pre-create model artifacts
        create_mock_model_artifacts(storage, DEFAULT_CONFIG)
        yield storage


@pytest.fixture
def client_with_mocks(temp_storage):
    """Create test client with mocked dependencies."""
    # Override dependencies
    app.dependency_overrides[get_storage] = lambda: temp_storage
    app.dependency_overrides[get_price_loader] = lambda: mock_price_loader

    client = TestClient(app)
    yield client

    # Cleanup
    app.dependency_overrides.clear()


# ============================================================================
# Scenario 1: Basic inference success
# ============================================================================


def test_inference_lstm_returns_200(client_with_mocks):
    """POST /inference/lstm with valid symbols returns 200."""
    response = client_with_mocks.post(
        "/inference/lstm",
        json={"symbols": ["AAPL", "MSFT"]},
    )
    assert response.status_code == 200


def test_inference_lstm_returns_predictions_for_all_symbols(client_with_mocks):
    """POST /inference/lstm returns predictions for all requested symbols."""
    response = client_with_mocks.post(
        "/inference/lstm",
        json={"symbols": ["AAPL", "MSFT", "GOOGL"]},
    )
    assert response.status_code == 200

    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 3

    symbols_returned = {p["symbol"] for p in data["predictions"]}
    assert symbols_returned == {"AAPL", "MSFT", "GOOGL"}


def test_inference_lstm_returns_required_fields(client_with_mocks):
    """POST /inference/lstm returns all required response fields."""
    response = client_with_mocks.post(
        "/inference/lstm",
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


def test_inference_lstm_returns_numeric_prediction(client_with_mocks):
    """POST /inference/lstm returns numeric predicted weekly return percentage."""
    response = client_with_mocks.post(
        "/inference/lstm",
        json={"symbols": ["AAPL"]},
    )
    assert response.status_code == 200

    data = response.json()
    pred = data["predictions"][0]

    assert pred["has_enough_history"] is True
    assert isinstance(pred["predicted_weekly_return_pct"], int | float)
    assert pred["direction"] in ["UP", "DOWN", "FLAT"]


# ============================================================================
# Scenario 2: Cutoff date always Friday + holiday-aware week boundaries
# ============================================================================


def test_inference_lstm_cutoff_always_friday(temp_storage):
    """Response as_of_date should always be a Friday, regardless of input."""
    from datetime import date as dt_date

    app.dependency_overrides.clear()
    app.dependency_overrides[get_storage] = lambda: temp_storage
    app.dependency_overrides[get_price_loader] = lambda: mock_price_loader

    client = TestClient(app)

    try:
        test_cases = [
            ("2026-01-12", "2026-01-09"),  # Monday -> Friday
            ("2026-01-09", "2026-01-02"),  # Friday -> prev Friday
            ("2026-01-10", "2026-01-09"),  # Saturday -> Friday
            ("2026-01-11", "2026-01-09"),  # Sunday -> Friday
            ("2026-01-14", "2026-01-09"),  # Wednesday -> Friday
        ]
        for input_date, expected_cutoff in test_cases:
            response = client.post(
                "/inference/lstm",
                json={"symbols": ["AAPL"], "as_of_date": input_date},
            )
            assert response.status_code == 200, f"Failed for input {input_date}"

            data = response.json()
            assert data["as_of_date"] == expected_cutoff, (
                f"Input {input_date}: expected {expected_cutoff}, got {data['as_of_date']}"
            )
            # Verify it's actually a Friday
            result_date = dt_date.fromisoformat(data["as_of_date"])
            assert result_date.weekday() == 4, f"Expected Friday for input {input_date}"
    finally:
        app.dependency_overrides.clear()


def test_inference_lstm_target_week_is_after_cutoff(temp_storage):
    """Target week should be the Mon-Fri AFTER the cutoff Friday."""
    app.dependency_overrides.clear()
    app.dependency_overrides[get_storage] = lambda: temp_storage
    app.dependency_overrides[get_price_loader] = lambda: mock_price_loader

    client = TestClient(app)

    try:
        # Saturday Jan 10 -> cutoff = Jan 9 (Fri) -> target week = Jan 12-16
        response = client.post(
            "/inference/lstm",
            json={"symbols": ["AAPL"], "as_of_date": "2026-01-10"},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["as_of_date"] == "2026-01-09"  # Friday cutoff
        assert data["target_week_start"] == "2026-01-12"  # Next Monday
        assert data["target_week_end"] == "2026-01-16"  # Next Friday
    finally:
        app.dependency_overrides.clear()


def test_inference_lstm_real_good_friday_holiday(temp_storage):
    """Test with actual exchange calendar for Good Friday 2025 (April 18)."""
    app.dependency_overrides.clear()
    app.dependency_overrides[get_storage] = lambda: temp_storage
    app.dependency_overrides[get_price_loader] = lambda: mock_price_loader

    client = TestClient(app)

    try:
        # Saturday Apr 12, 2025 -> cutoff = Apr 11 (Fri) -> target week = Apr 14-18
        # But Good Friday (Apr 18) is closed, so target_week_end should be Apr 17 (Thu)
        response = client.post(
            "/inference/lstm",
            json={"symbols": ["AAPL"], "as_of_date": "2025-04-12"},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["as_of_date"] == "2025-04-11"  # Friday cutoff
        assert data["target_week_start"] == "2025-04-14"  # Monday
        assert data["target_week_end"] == "2025-04-17"  # Thursday (Good Friday closed)
    finally:
        app.dependency_overrides.clear()


# ============================================================================
# Scenario 3: No current model
# ============================================================================


def test_inference_lstm_no_model_returns_503():
    """POST /inference/lstm returns 503 when no model is trained."""
    with tempfile.TemporaryDirectory() as tmpdir:
        empty_storage = LocalModelStorage(base_path=tmpdir)
        # Don't create any model artifacts

        app.dependency_overrides.clear()
        app.dependency_overrides[get_storage] = lambda: empty_storage
        app.dependency_overrides[get_price_loader] = lambda: mock_price_loader

        client = TestClient(app)

        try:
            response = client.post(
                "/inference/lstm",
                json={"symbols": ["AAPL"]},
            )
            assert response.status_code == 503
            assert "No trained LSTM model available" in response.json()["detail"]
        finally:
            app.dependency_overrides.clear()


# ============================================================================
# Scenario 4: Missing/insufficient data handling
# ============================================================================


def test_inference_lstm_missing_symbol_data(temp_storage):
    """Symbols with no price data should return has_enough_history=False."""
    app.dependency_overrides.clear()
    app.dependency_overrides[get_storage] = lambda: temp_storage
    app.dependency_overrides[get_price_loader] = lambda: mock_price_loader_no_data

    client = TestClient(app)

    try:
        response = client.post(
            "/inference/lstm",
            json={"symbols": ["UNKNOWNSYMBOL"]},
        )
        assert response.status_code == 200

        data = response.json()
        pred = data["predictions"][0]

        assert pred["symbol"] == "UNKNOWNSYMBOL"
        assert pred["has_enough_history"] is False
        assert pred["predicted_weekly_return_pct"] is None
    finally:
        app.dependency_overrides.clear()


# ============================================================================
# Scenario 5: Request validation
# ============================================================================


def test_inference_lstm_empty_symbols_returns_422(client_with_mocks):
    """POST /inference/lstm with empty symbols list returns 422."""
    response = client_with_mocks.post(
        "/inference/lstm",
        json={"symbols": []},
    )
    assert response.status_code == 422


def test_inference_lstm_no_symbols_returns_422(client_with_mocks):
    """POST /inference/lstm without symbols field returns 422."""
    response = client_with_mocks.post(
        "/inference/lstm",
        json={},
    )
    assert response.status_code == 422


def test_inference_lstm_custom_as_of_date(client_with_mocks):
    """POST /inference/lstm anchors custom as_of_date to Friday."""
    # 2025-01-06 is Monday -> cutoff should be 2025-01-03 (Friday)
    response = client_with_mocks.post(
        "/inference/lstm",
        json={"symbols": ["AAPL"], "as_of_date": "2025-01-06"},
    )
    assert response.status_code == 200

    data = response.json()
    # Monday Jan 6, 2025 -> Friday Jan 3, 2025
    assert data["as_of_date"] == "2025-01-03"


# ============================================================================
# Scenario 6: Sorting behavior (highest gain → highest loss)
# ============================================================================


def test_inference_lstm_predictions_sorted_by_return_desc(client_with_mocks):
    """POST /inference/lstm returns predictions sorted by predicted_weekly_return_pct descending."""
    response = client_with_mocks.post(
        "/inference/lstm",
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


def test_inference_lstm_insufficient_history_at_end(temp_storage):
    """Symbols with insufficient history should appear at the end of predictions."""

    def mock_price_loader_partial(symbols, start_date, end_date):
        """Return data only for some symbols."""
        prices = {}
        date_range = pd.bdate_range(start=start_date, end=end_date)

        # Only return data for AAPL and MSFT, not UNKNOWNSYMBOL
        for symbol in symbols:
            if symbol in ["AAPL", "MSFT"] and len(date_range) >= 10:
                np.random.seed(hash(symbol) % 2**32)
                base_price = 100.0
                returns = np.random.randn(len(date_range)) * 0.02
                prices_array = base_price * np.exp(np.cumsum(returns))

                df = pd.DataFrame(
                    {
                        "open": prices_array
                        * (1 + np.random.randn(len(date_range)) * 0.005),
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

    app.dependency_overrides.clear()
    app.dependency_overrides[get_storage] = lambda: temp_storage
    app.dependency_overrides[get_price_loader] = lambda: mock_price_loader_partial

    client = TestClient(app)

    try:
        response = client.post(
            "/inference/lstm",
            json={"symbols": ["UNKNOWNSYMBOL", "AAPL", "MSFT"]},
        )
        assert response.status_code == 200

        data = response.json()
        predictions = data["predictions"]

        # Should have 3 predictions
        assert len(predictions) == 3

        # UNKNOWNSYMBOL should be at the end (has_enough_history=False)
        assert predictions[-1]["symbol"] == "UNKNOWNSYMBOL"
        assert predictions[-1]["has_enough_history"] is False

        # AAPL and MSFT should be first two, sorted by return
        assert predictions[0]["has_enough_history"] is True
        assert predictions[1]["has_enough_history"] is True
    finally:
        app.dependency_overrides.clear()

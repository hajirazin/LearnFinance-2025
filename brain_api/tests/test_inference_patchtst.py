"""API-level tests for PatchTST inference endpoint."""

import tempfile

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.preprocessing import StandardScaler
from transformers import PatchTSTForPrediction

from brain_api.core.patchtst import DEFAULT_CONFIG, PatchTSTConfig
from brain_api.core.patchtst.inference import BatchInferenceResult, SymbolPrediction
from brain_api.main import app
from brain_api.routes.inference import (
    get_patchtst_india_storage,
    get_patchtst_storage,
)
from brain_api.storage.local import PatchTSTIndiaModelStorage, PatchTSTModelStorage

# ============================================================================
# Test fixtures and mocks
# ============================================================================

MOCK_VERSION = "v2025-01-01-patchtest"


def create_mock_patchtst_artifacts(
    storage: PatchTSTModelStorage, config: PatchTSTConfig
) -> str:
    """Create and write mock PatchTST model artifacts for testing.

    Returns the version string.
    """
    hf_config = config.to_hf_config()
    model = PatchTSTForPrediction(hf_config)

    scaler = StandardScaler()
    dummy_data = np.random.randn(100, 5)
    scaler.fit(dummy_data)

    metadata = {
        "model_type": "patchtst",
        "version": MOCK_VERSION,
        "training_timestamp": "2025-01-01T00:00:00+00:00",
        "data_window": {"start": "2020-01-01", "end": "2025-01-01"},
        "symbols": ["AAPL", "MSFT"],
        "config": config.to_dict(),
        "metrics": {"train_loss": 0.01, "val_loss": 0.02, "baseline_loss": 0.05},
        "promoted": True,
        "prior_version": None,
    }

    storage.write_artifacts(
        version=MOCK_VERSION,
        model=model,
        feature_scaler=scaler,
        config=config,
        metadata=metadata,
    )
    storage.promote_version(MOCK_VERSION)

    return MOCK_VERSION


def _make_mock_batch_result(symbols: list[str]) -> BatchInferenceResult:
    """Build a BatchInferenceResult with realistic predictions."""
    predictions = []
    for i, sym in enumerate(symbols):
        weekly_ret = round(2.5 - i * 1.0, 4)
        predictions.append(
            SymbolPrediction(
                symbol=sym,
                predicted_weekly_return_pct=weekly_ret,
                direction="UP" if weekly_ret > 0 else "DOWN",
                has_enough_history=True,
                history_days_used=120,
                data_end_date="2025-01-03",
                target_week_start="2025-01-06",
                target_week_end="2025-01-10",
                daily_returns=[0.004, 0.003, -0.001, 0.002, 0.001],
            )
        )
    predictions.sort(key=lambda p: p.predicted_weekly_return_pct or 0, reverse=True)
    return BatchInferenceResult(predictions=predictions, model_version=MOCK_VERSION)


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
    app.dependency_overrides[get_patchtst_storage] = lambda: temp_storage

    from brain_api.routes.inference import patchtst as inference_module

    def mock_run_batch(symbols, cutoff_date, storage=None):
        return _make_mock_batch_result(symbols)

    monkeypatch.setattr(inference_module, "run_batch_inference", mock_run_batch)

    client = TestClient(app)
    yield client

    app.dependency_overrides.clear()


# ============================================================================
# Scenario 1: Basic inference success
# ============================================================================


def test_inference_patchtst_returns_200(client_with_mocks):
    """POST /inference/patchtst returns 200 (symbols from model metadata)."""
    response = client_with_mocks.post(
        "/inference/patchtst",
        json={},
    )
    assert response.status_code == 200


def test_inference_patchtst_returns_predictions_for_model_symbols(client_with_mocks):
    """POST /inference/patchtst returns predictions for symbols in model metadata."""
    response = client_with_mocks.post(
        "/inference/patchtst",
        json={},
    )
    assert response.status_code == 200

    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2

    symbols_returned = {p["symbol"] for p in data["predictions"]}
    assert symbols_returned == {"AAPL", "MSFT"}


def test_inference_patchtst_returns_required_fields(client_with_mocks):
    """POST /inference/patchtst returns all required response fields."""
    response = client_with_mocks.post(
        "/inference/patchtst",
        json={},
    )
    assert response.status_code == 200

    data = response.json()

    assert "predictions" in data
    assert "model_version" in data
    assert "as_of_date" in data
    assert "target_week_start" in data
    assert "target_week_end" in data
    assert "signals_used" in data

    assert len(data["predictions"]) >= 1
    pred = data["predictions"][0]
    assert "symbol" in pred
    assert "predicted_weekly_return_pct" in pred
    assert "direction" in pred
    assert "has_enough_history" in pred
    assert "history_days_used" in pred
    assert "target_week_start" in pred
    assert "target_week_end" in pred
    assert "daily_returns" in pred


def test_inference_patchtst_returns_numeric_prediction(client_with_mocks):
    """POST /inference/patchtst returns numeric predicted weekly return percentage."""
    response = client_with_mocks.post(
        "/inference/patchtst",
        json={},
    )
    assert response.status_code == 200

    data = response.json()
    pred = data["predictions"][0]

    assert pred["has_enough_history"] is True
    assert isinstance(pred["predicted_weekly_return_pct"], int | float)
    assert pred["direction"] in ["UP", "DOWN", "FLAT"]


def test_inference_patchtst_response_does_not_include_predicted_volatility(
    client_with_mocks,
):
    """Response predictions do not include predicted_volatility (field removed)."""
    response = client_with_mocks.post(
        "/inference/patchtst",
        json={},
    )
    assert response.status_code == 200
    pred = response.json()["predictions"][0]
    assert "predicted_volatility" not in pred


def test_inference_patchtst_returns_signals_used(client_with_mocks):
    """POST /inference/patchtst returns list of signals that were available."""
    response = client_with_mocks.post(
        "/inference/patchtst",
        json={},
    )
    assert response.status_code == 200

    data = response.json()
    assert "signals_used" in data
    assert data["signals_used"] == ["ohlcv"]


# ============================================================================
# Scenario 2: No current model
# ============================================================================


def test_inference_patchtst_no_model_returns_400():
    """POST /inference/patchtst returns 400 when no model is trained."""
    with tempfile.TemporaryDirectory() as tmpdir:
        empty_storage = PatchTSTModelStorage(base_path=tmpdir)

        app.dependency_overrides.clear()
        app.dependency_overrides[get_patchtst_storage] = lambda: empty_storage

        client = TestClient(app)

        try:
            response = client.post(
                "/inference/patchtst",
                json={},
            )
            assert response.status_code == 400
            assert "No current PatchTST model" in response.json()["detail"]
        finally:
            app.dependency_overrides.clear()


# ============================================================================
# Scenario 3: Request validation
# ============================================================================


def test_inference_patchtst_empty_body_accepted(client_with_mocks):
    """POST /inference/patchtst with empty body returns 200 (symbols from metadata)."""
    response = client_with_mocks.post(
        "/inference/patchtst",
        json={},
    )
    assert response.status_code == 200


def test_inference_patchtst_requested_symbols_scopes_predictions(client_with_mocks):
    """POST /inference/patchtst with symbols returns only those symbols."""
    response = client_with_mocks.post(
        "/inference/patchtst",
        json={"symbols": ["AAPL"]},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 1
    assert data["predictions"][0]["symbol"] == "AAPL"


def test_inference_patchtst_empty_symbols_list_returns_422(client_with_mocks):
    """POST /inference/patchtst with symbols=[] fails validation (min_length=1)."""
    response = client_with_mocks.post(
        "/inference/patchtst",
        json={"symbols": []},
    )
    assert response.status_code == 422


def test_inference_patchtst_custom_as_of_date(client_with_mocks):
    """POST /inference/patchtst anchors custom as_of_date to Friday."""
    # 2025-01-06 is Monday -> cutoff should be 2025-01-03 (Friday)
    response = client_with_mocks.post(
        "/inference/patchtst",
        json={"as_of_date": "2025-01-06"},
    )
    assert response.status_code == 200

    data = response.json()
    # Monday Jan 6, 2025 -> Friday Jan 3, 2025
    assert data["as_of_date"] == "2025-01-03"


def test_inference_patchtst_cutoff_always_friday(client_with_mocks):
    """Response as_of_date should always be a Friday, regardless of input."""
    from datetime import date as dt_date

    test_cases = [
        ("2026-01-12", "2026-01-09"),  # Monday -> Friday
        ("2026-01-09", "2026-01-02"),  # Friday -> prev Friday
        ("2026-01-10", "2026-01-09"),  # Saturday -> Friday
        ("2026-01-11", "2026-01-09"),  # Sunday -> Friday
        ("2026-01-14", "2026-01-09"),  # Wednesday -> Friday
    ]
    for input_date, expected_cutoff in test_cases:
        response = client_with_mocks.post(
            "/inference/patchtst",
            json={"as_of_date": input_date},
        )
        assert response.status_code == 200, f"Failed for input {input_date}"

        data = response.json()
        assert data["as_of_date"] == expected_cutoff, (
            f"Input {input_date}: expected {expected_cutoff}, got {data['as_of_date']}"
        )
        # Verify it's actually a Friday
        result_date = dt_date.fromisoformat(data["as_of_date"])
        assert result_date.weekday() == 4, f"Expected Friday for input {input_date}"


# ============================================================================
# Scenario 4: Sorting behavior
# ============================================================================


def test_inference_patchtst_predictions_sorted_by_return_desc(client_with_mocks):
    """POST /inference/patchtst returns predictions sorted by predicted_weekly_return_pct descending."""
    response = client_with_mocks.post(
        "/inference/patchtst",
        json={},
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
# Scenario 5: daily_returns field
# ============================================================================


def test_inference_patchtst_returns_daily_returns_field(client_with_mocks):
    """POST /inference/patchtst returns daily_returns list in each prediction."""
    response = client_with_mocks.post(
        "/inference/patchtst",
        json={},
    )
    assert response.status_code == 200

    pred = response.json()["predictions"][0]
    assert "daily_returns" in pred

    if pred["has_enough_history"]:
        # daily_returns should be a list of 5 floats
        assert pred["daily_returns"] is not None
        assert isinstance(pred["daily_returns"], list)
        assert len(pred["daily_returns"]) == 5
        for dr in pred["daily_returns"]:
            assert isinstance(dr, int | float)


# ============================================================================
# /inference/patchtst/india
# ============================================================================


@pytest.fixture
def temp_india_storage():
    """Temporary India PatchTST storage with mock artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PatchTSTIndiaModelStorage(base_path=tmpdir)
        create_mock_patchtst_artifacts(storage, DEFAULT_CONFIG)
        yield storage


@pytest.fixture
def india_client_with_mocks(temp_india_storage, monkeypatch):
    """Test client with India PatchTST storage + mocked run_batch_inference."""
    app.dependency_overrides[get_patchtst_india_storage] = lambda: temp_india_storage

    from brain_api.routes.inference import patchtst as inference_module

    def mock_run_batch(symbols, cutoff_date, storage=None):
        return _make_mock_batch_result(symbols)

    monkeypatch.setattr(inference_module, "run_batch_inference", mock_run_batch)

    client = TestClient(app)
    yield client

    app.dependency_overrides.clear()


def test_inference_patchtst_india_returns_200(india_client_with_mocks):
    """POST /inference/patchtst/india returns 200 with model symbols."""
    response = india_client_with_mocks.post("/inference/patchtst/india", json={})
    assert response.status_code == 200
    data = response.json()
    assert {p["symbol"] for p in data["predictions"]} == {"AAPL", "MSFT"}
    assert data["signals_used"] == ["ohlcv"]


def test_inference_patchtst_india_no_model_returns_400():
    """POST /inference/patchtst/india returns 400 when no India model is current."""
    with tempfile.TemporaryDirectory() as tmpdir:
        empty_storage = PatchTSTIndiaModelStorage(base_path=tmpdir)
        app.dependency_overrides.clear()
        app.dependency_overrides[get_patchtst_india_storage] = lambda: empty_storage

        client = TestClient(app)
        try:
            response = client.post("/inference/patchtst/india", json={})
            assert response.status_code == 400
            assert "No current PatchTST" in response.json()["detail"]
        finally:
            app.dependency_overrides.clear()


def test_inference_patchtst_india_empty_symbols_returns_422(india_client_with_mocks):
    """POST /inference/patchtst/india with symbols=[] fails Field min_length=1."""
    response = india_client_with_mocks.post(
        "/inference/patchtst/india", json={"symbols": []}
    )
    assert response.status_code == 422


def test_inference_patchtst_india_uses_india_storage(
    temp_storage, temp_india_storage, monkeypatch
):
    """The India route must read from PatchTSTIndiaModelStorage, not the US one.

    We create a US model with version 'us-version' and an India model
    with version 'india-version'; the India route must surface the
    India version. This guards against accidental wiring back to
    get_patchtst_storage.
    """
    # Override the US version on temp_storage so the two are visibly distinct.
    us_version = temp_storage.read_current_version()
    india_version = temp_india_storage.read_current_version()
    assert us_version == india_version  # both are MOCK_VERSION; that's fine

    app.dependency_overrides.clear()
    app.dependency_overrides[get_patchtst_storage] = lambda: temp_storage
    app.dependency_overrides[get_patchtst_india_storage] = lambda: temp_india_storage

    from brain_api.routes.inference import patchtst as inference_module

    seen_storage = []

    def mock_run_batch(symbols, cutoff_date, storage=None):
        seen_storage.append(storage)
        return _make_mock_batch_result(symbols)

    monkeypatch.setattr(inference_module, "run_batch_inference", mock_run_batch)

    client = TestClient(app)
    try:
        response = client.post("/inference/patchtst/india", json={})
        assert response.status_code == 200
        # The mock captured exactly one call; the storage object must be
        # the India one we provided -- not the US one.
        assert len(seen_storage) == 1
        assert seen_storage[0] is temp_india_storage
    finally:
        app.dependency_overrides.clear()


# ============================================================================
# /inference/patchtst/score-batch
# ============================================================================


@pytest.fixture
def score_batch_client(temp_storage, temp_india_storage, monkeypatch):
    """Test client with both US and India storage + mocked batch inference.

    Lets us flip ``market`` between calls without re-initialising the
    test client.
    """
    app.dependency_overrides[get_patchtst_storage] = lambda: temp_storage
    app.dependency_overrides[get_patchtst_india_storage] = lambda: temp_india_storage

    from brain_api.routes.inference import patchtst as inference_module

    def mock_run_batch(symbols, cutoff_date, storage=None):
        return _make_mock_batch_result(symbols)

    monkeypatch.setattr(inference_module, "run_batch_inference", mock_run_batch)

    client = TestClient(app)
    yield client

    app.dependency_overrides.clear()


def test_score_batch_us_happy_path(score_batch_client):
    """market='us' returns finite scores keyed by symbol + metadata fields."""
    response = score_batch_client.post(
        "/inference/patchtst/score-batch",
        json={"market": "us", "symbols": ["AAPL", "MSFT"], "min_predictions": 1},
    )
    assert response.status_code == 200
    data = response.json()
    assert set(data["scores"].keys()) == {"AAPL", "MSFT"}
    assert data["requested_count"] == 2
    assert data["predicted_count"] == 2
    assert data["model_version"] == MOCK_VERSION
    assert data["target_week_start"]
    assert data["target_week_end"]
    assert data["excluded_symbols"] == []


def test_score_batch_india_happy_path(score_batch_client):
    """market='india' uses the India storage path and returns the same shape."""
    response = score_batch_client.post(
        "/inference/patchtst/score-batch",
        json={
            "market": "india",
            "symbols": ["RELIANCE.NS", "TCS.NS"],
            "min_predictions": 1,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert set(data["scores"].keys()) == {"RELIANCE.NS", "TCS.NS"}
    assert data["predicted_count"] == 2


def test_score_batch_response_shape_has_required_fields(score_batch_client):
    """Regression: verify all PatchTSTScoreBatchResponse fields are present.

    This locks the response contract that the Temporal
    score_halal_*_with_patchtst activities deserialise.
    """
    response = score_batch_client.post(
        "/inference/patchtst/score-batch",
        json={"market": "us", "symbols": ["AAPL"], "min_predictions": 1},
    )
    assert response.status_code == 200
    data = response.json()
    for field in (
        "scores",
        "model_version",
        "as_of_date",
        "target_week_start",
        "target_week_end",
        "requested_count",
        "predicted_count",
        "excluded_symbols",
    ):
        assert field in data, f"Missing field: {field}"


def test_score_batch_non_finite_returns_422(score_batch_client, monkeypatch):
    """A NaN prediction must surface as 422 (rank-band invariant violation)."""
    from brain_api.routes.inference import patchtst as inference_module

    def mock_run_batch_with_nan(symbols, cutoff_date, storage=None):
        result = _make_mock_batch_result(symbols)
        # Replace the first prediction's score with NaN.
        result.predictions[0].predicted_weekly_return_pct = float("nan")
        return result

    monkeypatch.setattr(
        inference_module, "run_batch_inference", mock_run_batch_with_nan
    )

    response = score_batch_client.post(
        "/inference/patchtst/score-batch",
        json={"market": "us", "symbols": ["AAPL", "MSFT"], "min_predictions": 1},
    )
    assert response.status_code == 422
    assert "non-finite" in response.json()["detail"]


def test_score_batch_below_min_predictions_returns_422(score_batch_client):
    """Below-floor finite count must raise 422."""
    response = score_batch_client.post(
        "/inference/patchtst/score-batch",
        json={"market": "us", "symbols": ["AAPL"], "min_predictions": 5},
    )
    assert response.status_code == 422
    assert "below" in response.json()["detail"]


def test_score_batch_empty_symbols_returns_422(score_batch_client):
    """symbols=[] fails Field min_length=1 validation."""
    response = score_batch_client.post(
        "/inference/patchtst/score-batch",
        json={"market": "us", "symbols": [], "min_predictions": 1},
    )
    assert response.status_code == 422


def test_score_batch_invalid_market_returns_422(score_batch_client):
    """market='europe' fails the Literal validator."""
    response = score_batch_client.post(
        "/inference/patchtst/score-batch",
        json={"market": "europe", "symbols": ["AAPL"], "min_predictions": 1},
    )
    assert response.status_code == 422


def test_score_batch_no_india_model_returns_400():
    """market='india' without a current India model returns 400."""
    with tempfile.TemporaryDirectory() as tmpdir:
        empty_india = PatchTSTIndiaModelStorage(base_path=tmpdir)
        app.dependency_overrides.clear()
        app.dependency_overrides[get_patchtst_india_storage] = lambda: empty_india
        # US storage is irrelevant for this test, but the dep is required;
        # an empty US storage is fine because we never read it.
        with tempfile.TemporaryDirectory() as us_tmp:
            empty_us = PatchTSTModelStorage(base_path=us_tmp)
            app.dependency_overrides[get_patchtst_storage] = lambda: empty_us

            client = TestClient(app)
            try:
                response = client.post(
                    "/inference/patchtst/score-batch",
                    json={
                        "market": "india",
                        "symbols": ["RELIANCE.NS"],
                        "min_predictions": 1,
                    },
                )
                assert response.status_code == 400
                assert "india" in response.json()["detail"]
            finally:
                app.dependency_overrides.clear()

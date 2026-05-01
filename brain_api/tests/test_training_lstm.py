"""API-level tests for LSTM training endpoint.

The endpoint resolves training symbols and storage from the per-bucket
registry (``brain_api.core.model_buckets``); these tests inject a
temporary in-memory bucket override so the production registry stays
untouched while still exercising the full request/response path.
"""

import os
import tempfile
from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.preprocessing import StandardScaler

import brain_api.routes.training.lstm as lstm_route
from brain_api.core.lstm import DatasetResult, LSTMModel, TrainingResult
from brain_api.core.model_buckets import ModelType, get_bucket
from brain_api.main import app
from brain_api.routes.training import (
    get_dataset_builder,
    get_price_loader,
    get_trainer,
)
from brain_api.storage.forecaster_snapshots import SnapshotLocalStorage
from brain_api.storage.lstm.local import LSTMHalalNewModelStorage

# Default URL for tests that exercise the *main* training path. Snapshot
# backfill is intentionally disabled here -- the backfill branch is
# covered explicitly by the dedicated ``test_train_lstm_skip_snapshot_*``
# pair below, which monkeypatches the heavy compute helpers so the loop
# runs end-to-end against deterministic stubs.
TRAIN_URL_NO_SNAPSHOT = "/train/lstm?skip_snapshot=true"

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


def mock_dataset_builder(prices, config) -> DatasetResult:
    """Return a mock dataset result for direct 5-day close-return prediction."""
    n_samples = 10
    return DatasetResult(
        X=np.random.randn(n_samples, config.sequence_length, config.input_size),
        y=np.random.randn(n_samples, 5),
        feature_scaler=StandardScaler(),
    )


def mock_trainer(X, y, feature_scaler, config, shutdown_event=None) -> TrainingResult:
    """Return a mock training result with controllable metrics."""
    model = LSTMModel(config)
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
    model = LSTMModel(config)
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
        yield LSTMHalalNewModelStorage(base_path=tmpdir)


def _override_lstm_halal_new_bucket(monkeypatch, temp_storage, symbols_fn=mock_symbols):
    """Swap the ``(LSTM, halal_new)`` registry entry to use the test
    storage and symbol resolver. Restored automatically by ``monkeypatch``.

    Symbol resolution + storage instantiation now happen inside the
    endpoint via the bucket registry (the old ``Depends(get_storage)``
    seam was deliberately removed so two parallel workflows can hit the
    same endpoint with different universes). Tests therefore mutate the
    bucket itself; the override is per-test thanks to ``monkeypatch``.
    """
    from brain_api.core import model_buckets

    original = get_bucket(ModelType.LSTM, "halal_new")
    patched = replace(
        original,
        local_storage_class=lambda: temp_storage,
        symbols_resolver=symbols_fn,
    )
    monkeypatch.setitem(
        model_buckets._BUCKETS,
        (ModelType.LSTM, "halal_new"),
        patched,
    )


def _patch_backfill_internals(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the heavy compute helpers used by ``_backfill_lstm_snapshots``.

    The route imports ``load_prices_yfinance`` / ``build_dataset`` /
    ``train_model_pytorch`` at module top, so the only safe seam is to
    monkeypatch the rebound names on
    :mod:`brain_api.routes.training.lstm` itself; the existing
    ``Depends(get_price_loader)`` / ``get_dataset_builder`` /
    ``get_trainer`` overrides only cover the *main* training path.

    Mocks preserve return shapes (``DatasetResult`` / ``TrainingResult``)
    so the backfill loop's snapshot-write + HF-upload-gate code still
    runs end-to-end -- only the network/PyTorch wall-clock cost is
    removed (per AGENTS.md rule: mock side effects, never skip them).
    """
    monkeypatch.setattr(lstm_route, "load_prices_yfinance", mock_price_loader)
    monkeypatch.setattr(
        lstm_route,
        "build_dataset",
        lambda prices, config: mock_dataset_builder(prices, config),
    )
    monkeypatch.setattr(
        lstm_route,
        "train_model_pytorch",
        lambda X, y, feature_scaler, config, shutdown_event=None: mock_trainer(
            X, y, feature_scaler, config, shutdown_event=shutdown_event
        ),
    )


@pytest.fixture
def client_with_mocks(temp_storage, monkeypatch):
    """Create test client with mocked dependencies for the *main* training path.

    Tests that use this fixture POST to :data:`TRAIN_URL_NO_SNAPSHOT`
    (``?skip_snapshot=true``) so they only exercise the main training
    flow -- not the snapshot backfill, which has its own dedicated
    branch tests below.
    """
    _override_lstm_halal_new_bucket(monkeypatch, temp_storage)
    app.dependency_overrides[get_price_loader] = lambda: mock_price_loader
    app.dependency_overrides[get_dataset_builder] = lambda: mock_dataset_builder
    app.dependency_overrides[get_trainer] = lambda: mock_trainer

    os.environ["LSTM_TRAIN_LOOKBACK_YEARS"] = "10"
    os.environ["LSTM_TRAIN_WINDOW_END_DATE"] = "2025-01-01"

    client = TestClient(app)
    yield client

    app.dependency_overrides.clear()
    os.environ.pop("LSTM_TRAIN_LOOKBACK_YEARS", None)
    os.environ.pop("LSTM_TRAIN_WINDOW_END_DATE", None)


@pytest.fixture
def client_with_backfill_mocks(temp_storage, monkeypatch):
    """Test client with ALL training paths mocked, including snapshot backfill.

    Mirrors :func:`client_with_mocks` but also patches the route's
    module-level ``load_prices_yfinance`` / ``build_dataset`` /
    ``train_model_pytorch`` so the ``skip_snapshot=False`` branch runs
    in milliseconds while still exercising the real backfill control
    flow (snapshot existence checks, on-disk writes via
    ``SnapshotLocalStorage``, HF upload gating, ``gc.collect`` /
    MPS cache cleanup).
    """
    _override_lstm_halal_new_bucket(monkeypatch, temp_storage)
    app.dependency_overrides[get_price_loader] = lambda: mock_price_loader
    app.dependency_overrides[get_dataset_builder] = lambda: mock_dataset_builder
    app.dependency_overrides[get_trainer] = lambda: mock_trainer
    _patch_backfill_internals(monkeypatch)

    # Use a 2-year lookback (instead of the production-realistic 10) so
    # the backfill loop in ``_backfill_lstm_snapshots`` runs 2
    # iterations instead of 11. Each iteration still does a real
    # ``LSTMModel(config)`` construction + ``torch.save`` + scaler
    # pickle + metadata JSON (the *side effects* the test must
    # exercise), so we don't lose any coverage -- the ``len(snapshots)
    # > 1`` assertion still holds (end-date 2024-12-27 + 2 backfill
    # cutoffs). Keeps the snapshot-write test under ~300 ms.
    os.environ["LSTM_TRAIN_LOOKBACK_YEARS"] = "2"
    os.environ["LSTM_TRAIN_WINDOW_END_DATE"] = "2025-01-01"

    client = TestClient(app)
    yield client

    app.dependency_overrides.clear()
    os.environ.pop("LSTM_TRAIN_LOOKBACK_YEARS", None)
    os.environ.pop("LSTM_TRAIN_WINDOW_END_DATE", None)


# ============================================================================
# Scenario 1: Empty-body success + resolved window
# ============================================================================


def test_train_lstm_empty_body_returns_202(client_with_mocks):
    """POST /train/lstm with empty body returns 202 on first call."""
    response = client_with_mocks.post(TRAIN_URL_NO_SNAPSHOT, json={})
    assert response.status_code == 202


def test_train_lstm_no_body_returns_202(client_with_mocks):
    """POST /train/lstm with no body returns 202 on first call."""
    response = client_with_mocks.post(TRAIN_URL_NO_SNAPSHOT)
    assert response.status_code == 202


def test_train_lstm_explicit_universe_returns_202(client_with_mocks):
    """POST /train/lstm with explicit universe field returns 202 on first call."""
    response = client_with_mocks.post(
        TRAIN_URL_NO_SNAPSHOT, json={"universe": "halal_new"}
    )
    assert response.status_code == 202


def test_train_lstm_unknown_universe_returns_422(client_with_mocks):
    """Unknown universe in the request body must be rejected with 422."""
    response = client_with_mocks.post(
        TRAIN_URL_NO_SNAPSHOT, json={"universe": "not_a_universe"}
    )
    assert response.status_code == 422
    assert "not_a_universe" in response.text


def test_train_lstm_returns_resolved_window(client_with_mocks):
    """POST /train/lstm returns Friday-anchored data_window_end from config."""
    response1 = client_with_mocks.post(TRAIN_URL_NO_SNAPSHOT, json={})
    assert response1.status_code == 202

    response = client_with_mocks.post(TRAIN_URL_NO_SNAPSHOT, json={})
    assert response.status_code == 200

    data = response.json()
    assert "data_window_start" in data
    assert "data_window_end" in data
    assert data["data_window_end"] == "2024-12-27"
    assert data["data_window_start"] == "2014-01-01"


def test_train_lstm_returns_required_fields(client_with_mocks):
    """POST /train/lstm returns all required response fields."""
    response1 = client_with_mocks.post(TRAIN_URL_NO_SNAPSHOT, json={})
    assert response1.status_code == 202

    response = client_with_mocks.post(TRAIN_URL_NO_SNAPSHOT, json={})
    assert response.status_code == 200

    data = response.json()
    assert "version" in data
    assert "data_window_start" in data
    assert "data_window_end" in data
    assert "metrics" in data
    assert "promoted" in data

    assert isinstance(data["metrics"], dict)


# ============================================================================
# Scenario 2: Idempotency on rerun
# ============================================================================


def test_train_lstm_idempotent_version(client_with_mocks):
    """Calling POST /train/lstm twice returns the same version."""
    response1 = client_with_mocks.post(TRAIN_URL_NO_SNAPSHOT, json={})
    assert response1.status_code == 202

    response2 = client_with_mocks.post(TRAIN_URL_NO_SNAPSHOT, json={})
    assert response2.status_code == 200
    version2 = response2.json()["version"]

    response3 = client_with_mocks.post(TRAIN_URL_NO_SNAPSHOT, json={})
    assert response3.status_code == 200
    version3 = response3.json()["version"]

    assert version2 == version3, "Version should be identical on rerun with same config"


def test_train_lstm_idempotent_does_not_change_current(client_with_mocks, temp_storage):
    """Rerunning training does not change 'current' pointer if already promoted."""
    response1 = client_with_mocks.post(TRAIN_URL_NO_SNAPSHOT, json={})
    assert response1.status_code == 202

    response2 = client_with_mocks.post(TRAIN_URL_NO_SNAPSHOT, json={})
    assert response2.status_code == 200
    version2 = response2.json()["version"]

    current_after_first = temp_storage.read_current_version()

    response3 = client_with_mocks.post(TRAIN_URL_NO_SNAPSHOT, json={})
    assert response3.status_code == 200
    version3 = response3.json()["version"]

    current_after_second = temp_storage.read_current_version()

    assert version2 == version3
    assert current_after_first == current_after_second


# ============================================================================
# Scenario 3: Promotion gate behavior
# ============================================================================


def test_train_lstm_first_model_always_promoted(client_with_mocks):
    """First model is always promoted (no prior model to compare against)."""
    response1 = client_with_mocks.post(TRAIN_URL_NO_SNAPSHOT, json={})
    assert response1.status_code == 202

    response2 = client_with_mocks.post(TRAIN_URL_NO_SNAPSHOT, json={})
    assert response2.status_code == 200

    data = response2.json()
    assert data["promoted"] is True


def test_train_lstm_not_promoted_when_worse_than_prior(monkeypatch):
    """Model is NOT promoted when worse than prior model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fresh_storage = LSTMHalalNewModelStorage(base_path=tmpdir)

        app.dependency_overrides.clear()

        _override_lstm_halal_new_bucket(monkeypatch, fresh_storage)
        app.dependency_overrides[get_price_loader] = lambda: mock_price_loader
        app.dependency_overrides[get_dataset_builder] = lambda: mock_dataset_builder
        app.dependency_overrides[get_trainer] = lambda: mock_trainer

        os.environ["LSTM_TRAIN_LOOKBACK_YEARS"] = "10"
        os.environ["LSTM_TRAIN_WINDOW_END_DATE"] = "2025-06-15"

        client = TestClient(app)

        try:
            response1 = client.post(TRAIN_URL_NO_SNAPSHOT, json={})
            assert response1.status_code == 202

            response2 = client.post(TRAIN_URL_NO_SNAPSHOT, json={})
            assert response2.status_code == 200
            first_version = response2.json()["version"]
            assert fresh_storage.read_current_version() == first_version

            app.dependency_overrides[get_trainer] = (
                lambda: mock_trainer_worse_than_baseline
            )
            os.environ["LSTM_TRAIN_WINDOW_END_DATE"] = "2025-06-23"

            response3 = client.post(TRAIN_URL_NO_SNAPSHOT, json={})
            assert response3.status_code == 202

            response4 = client.post(TRAIN_URL_NO_SNAPSHOT, json={})
            assert response4.status_code == 200

            data = response4.json()
            assert data["promoted"] is False

            current = fresh_storage.read_current_version()
            assert current == first_version
        finally:
            app.dependency_overrides.clear()
            os.environ.pop("LSTM_TRAIN_LOOKBACK_YEARS", None)
            os.environ.pop("LSTM_TRAIN_WINDOW_END_DATE", None)


def test_train_lstm_current_unchanged_when_not_promoted(temp_storage, monkeypatch):
    """The 'current' pointer is unchanged when promotion fails."""
    app.dependency_overrides.clear()

    _override_lstm_halal_new_bucket(monkeypatch, temp_storage)
    app.dependency_overrides[get_price_loader] = lambda: mock_price_loader
    app.dependency_overrides[get_dataset_builder] = lambda: mock_dataset_builder
    app.dependency_overrides[get_trainer] = lambda: mock_trainer

    os.environ["LSTM_TRAIN_LOOKBACK_YEARS"] = "10"
    os.environ["LSTM_TRAIN_WINDOW_END_DATE"] = "2025-01-01"

    client = TestClient(app)

    response1 = client.post(TRAIN_URL_NO_SNAPSHOT, json={})
    assert response1.status_code == 202

    response2 = client.post(TRAIN_URL_NO_SNAPSHOT, json={})
    assert response2.status_code == 200
    promoted_version = response2.json()["version"]

    current_before = temp_storage.read_current_version()
    assert current_before == promoted_version

    app.dependency_overrides[get_trainer] = lambda: mock_trainer_worse_than_baseline
    os.environ["LSTM_TRAIN_WINDOW_END_DATE"] = "2025-01-13"

    response3 = client.post(TRAIN_URL_NO_SNAPSHOT, json={})
    assert response3.status_code == 202

    response4 = client.post(TRAIN_URL_NO_SNAPSHOT, json={})
    assert response4.status_code == 200
    data4 = response4.json()

    assert data4["promoted"] is False

    current_after = temp_storage.read_current_version()
    assert current_after == promoted_version

    app.dependency_overrides.clear()
    os.environ.pop("LSTM_TRAIN_LOOKBACK_YEARS", None)
    os.environ.pop("LSTM_TRAIN_WINDOW_END_DATE", None)


# ============================================================================
# Scenario 4: skip_snapshot branch coverage
#
# These two tests are the only place in the file that exercises
# ``_backfill_lstm_snapshots``'s control flow. The bulk of the tests
# above pass ``?skip_snapshot=true`` for speed; here we explicitly
# assert both branches against on-disk state.
# ============================================================================


def _wait_for_terminal_response(
    client: TestClient, url: str, max_attempts: int = 5
) -> dict:
    """Drive the route until the BackgroundTask completes (200 not 202).

    The route returns 202 while the job is in flight and 200 once the
    artifact is on disk. Two POSTs is the documented pattern in the
    rest of this file; loop a couple extra times to be robust against
    the BackgroundTask running slightly slower under the new mock
    surface.
    """
    last_response = None
    for _ in range(max_attempts):
        last_response = client.post(url, json={})
        if last_response.status_code == 200:
            return last_response.json()
    raise AssertionError(
        f"Training did not converge to 200 within {max_attempts} POSTs. "
        f"Last status: {last_response.status_code if last_response else 'n/a'}"
    )


def test_train_lstm_skip_snapshot_true_writes_no_snapshot(client_with_mocks):
    """``?skip_snapshot=true`` must NOT create any snapshot on disk.

    The conftest ``isolate_forecaster_snapshots`` fixture redirects
    ``SnapshotLocalStorage`` writes to a per-test tmp dir, so this
    assertion is a closed-loop check on the route's branch behavior:
    nothing is written when the operator passes the skip flag.
    """
    _ = _wait_for_terminal_response(client_with_mocks, TRAIN_URL_NO_SNAPSHOT)

    snapshot_storage = SnapshotLocalStorage("lstm_halal_new")
    assert snapshot_storage.list_snapshots() == [], (
        "skip_snapshot=true must not write any snapshots, but found: "
        f"{snapshot_storage.list_snapshots()}"
    )


def test_train_lstm_skip_snapshot_false_writes_snapshots(client_with_backfill_mocks):
    """Default (``skip_snapshot=false``) writes the end-date snapshot AND backfills history.

    Uses :func:`client_with_backfill_mocks` so the heavy compute inside
    ``_backfill_lstm_snapshots`` (yfinance download + per-year LSTM
    fit) is replaced by deterministic stubs while the real on-disk
    write + existence-check logic still runs. Asserts on the snapshot
    directory the route created under the autouse tmp path.
    """
    _ = _wait_for_terminal_response(client_with_backfill_mocks, "/train/lstm")

    snapshot_storage = SnapshotLocalStorage("lstm_halal_new")
    snapshots = snapshot_storage.list_snapshots()
    assert snapshots, (
        "skip_snapshot=false must write at least one snapshot, but on-disk "
        "snapshot list was empty."
    )
    end_date_iso = "2024-12-27"
    end_date_snapshots = [s for s in snapshots if s.isoformat() == end_date_iso]
    assert end_date_snapshots, (
        f"Expected end-date snapshot {end_date_iso} on disk, got: "
        f"{[s.isoformat() for s in snapshots]}"
    )
    assert len(snapshots) > 1, (
        "Backfill must populate historical snapshots in addition to the "
        f"end-date one. Got only: {[s.isoformat() for s in snapshots]}"
    )

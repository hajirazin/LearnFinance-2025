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
        best_epoch=1,
        stopped_epoch=1,
    )


def mock_trainer_worse_than_baseline(
    X, y, feature_scaler, config, shutdown_event=None
) -> TrainingResult:
    """Return a mock training result that is worse than baseline.

    Under the always-promote-with-guardrails policy this still passes
    every guardrail (all metrics finite + positive, all artifact files
    written), so it is now PROMOTED. Tests that used to assert the
    opposite were testing the prior-comparison gate, which has been
    removed in favor of guardrails on the new artifact only.
    """
    model = LSTMModel(config)
    return TrainingResult(
        model=model,
        feature_scaler=feature_scaler if feature_scaler else StandardScaler(),
        config=config,
        train_loss=0.10,
        val_loss=0.10,
        baseline_loss=0.05,
        best_epoch=1,
        stopped_epoch=1,
    )


def mock_trainer_nan_val_loss(
    X, y, feature_scaler, config, shutdown_event=None
) -> TrainingResult:
    """Mock trainer that returns ``NaN`` val_loss to trip the guardrail.

    The forecaster artifact health check rejects any non-finite metric;
    this fixture is the canonical way to exercise that branch from the
    HTTP layer.
    """
    model = LSTMModel(config)
    return TrainingResult(
        model=model,
        feature_scaler=feature_scaler if feature_scaler else StandardScaler(),
        config=config,
        train_loss=0.01,
        val_loss=float("nan"),
        baseline_loss=0.05,
        best_epoch=1,
        stopped_epoch=1,
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

    Backfill mechanics now live in
    :mod:`brain_api.routes.training.snapshot_phase` (extracted to keep
    ``lstm.py`` under the AGENTS.md 600-line file ceiling). The
    backfill loop imports ``load_prices_yfinance`` / ``build_dataset``
    / ``train_model_pytorch`` at that module's top, so the only safe
    seam is to monkeypatch those rebound names on
    ``snapshot_phase``; the existing
    ``Depends(get_price_loader)`` / ``get_dataset_builder`` /
    ``get_trainer`` overrides only cover the *main* training path.

    Mocks preserve return shapes (``DatasetResult`` / ``TrainingResult``)
    so the backfill loop's snapshot-write + HF-upload-gate code still
    runs end-to-end -- only the network/PyTorch wall-clock cost is
    removed (per AGENTS.md rule: mock side effects, never skip them).
    """
    from brain_api.routes.training import snapshot_phase

    monkeypatch.setattr(snapshot_phase, "load_prices_yfinance", mock_price_loader)
    monkeypatch.setattr(
        snapshot_phase,
        "build_dataset",
        lambda prices, config: mock_dataset_builder(prices, config),
    )
    monkeypatch.setattr(
        snapshot_phase,
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


def test_train_lstm_returns_required_fields(client_with_mocks, temp_storage):
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
    assert data["metrics"]["best_epoch"] == 1
    assert data["metrics"]["stopped_epoch"] == 1
    on_disk = temp_storage.read_metadata(data["version"])
    assert on_disk is not None
    assert on_disk["metrics"]["best_epoch"] == 1
    assert on_disk["metrics"]["stopped_epoch"] == 1


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


def test_train_lstm_promotes_even_when_worse_than_prior(monkeypatch):
    """Always-promote policy: a healthy model promotes regardless of prior val_loss.

    Pre-refactor this test asserted ``promoted is False``; the
    universe-drift critique made that comparison meaningless. Now the
    only thing the gate cares about is the new artifact's own health,
    so a healthy run with worse-than-prior val_loss MUST still
    promote and bump the ``current`` pointer.
    """
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

            # Second run with WORSE val_loss than the prior. Under the
            # old gate this would not promote; under guardrails it does.
            app.dependency_overrides[get_trainer] = lambda: (
                mock_trainer_worse_than_baseline
            )
            os.environ["LSTM_TRAIN_WINDOW_END_DATE"] = "2025-06-23"

            response3 = client.post(TRAIN_URL_NO_SNAPSHOT, json={})
            assert response3.status_code == 202

            response4 = client.post(TRAIN_URL_NO_SNAPSHOT, json={})
            assert response4.status_code == 200

            data = response4.json()
            assert data["promoted"] is True

            current = fresh_storage.read_current_version()
            assert current == data["version"]
            assert current != first_version
        finally:
            app.dependency_overrides.clear()
            os.environ.pop("LSTM_TRAIN_LOOKBACK_YEARS", None)
            os.environ.pop("LSTM_TRAIN_WINDOW_END_DATE", None)


def test_train_lstm_not_promoted_when_val_loss_is_nan(temp_storage, monkeypatch):
    """The new guardrail rejects NaN val_loss and leaves ``current`` unchanged.

    This is the canonical example of "the gate now cares about the new
    artifact's health, not its relation to a prior". A trainer that
    silently diverges to NaN must NOT ship; ``current`` stays on the
    prior healthy version (or stays absent if there was no prior).
    """
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
    healthy_version = response2.json()["version"]
    assert temp_storage.read_current_version() == healthy_version

    # Second run produces NaN val_loss -- guardrail must reject.
    app.dependency_overrides[get_trainer] = lambda: mock_trainer_nan_val_loss
    os.environ["LSTM_TRAIN_WINDOW_END_DATE"] = "2025-01-13"

    response3 = client.post(TRAIN_URL_NO_SNAPSHOT, json={})
    assert response3.status_code == 202

    response4 = client.post(TRAIN_URL_NO_SNAPSHOT, json={})
    assert response4.status_code == 200
    data4 = response4.json()

    assert data4["promoted"] is False
    # ``current`` must stay on the prior healthy version.
    assert temp_storage.read_current_version() == healthy_version

    app.dependency_overrides.clear()
    os.environ.pop("LSTM_TRAIN_LOOKBACK_YEARS", None)
    os.environ.pop("LSTM_TRAIN_WINDOW_END_DATE", None)


def test_train_lstm_idempotent_rerun_returns_cached_failure_reasons(
    temp_storage, monkeypatch
):
    """A rerun of an unhealthy version returns the same failure_reasons
    from the cached metadata.json -- proves the field round-trips."""
    app.dependency_overrides.clear()

    _override_lstm_halal_new_bucket(monkeypatch, temp_storage)
    app.dependency_overrides[get_price_loader] = lambda: mock_price_loader
    app.dependency_overrides[get_dataset_builder] = lambda: mock_dataset_builder
    app.dependency_overrides[get_trainer] = lambda: mock_trainer_nan_val_loss

    os.environ["LSTM_TRAIN_LOOKBACK_YEARS"] = "10"
    os.environ["LSTM_TRAIN_WINDOW_END_DATE"] = "2025-01-01"

    client = TestClient(app)

    try:
        response1 = client.post(TRAIN_URL_NO_SNAPSHOT, json={})
        assert response1.status_code == 202

        response2 = client.post(TRAIN_URL_NO_SNAPSHOT, json={})
        assert response2.status_code == 200
        first_data = response2.json()
        assert first_data["promoted"] is False

        # Idempotent rerun -- same window, same trainer -> 200 with the
        # cached metadata. metadata.json is the source of truth on disk.
        response3 = client.post(TRAIN_URL_NO_SNAPSHOT, json={})
        assert response3.status_code == 200
        rerun_data = response3.json()
        assert rerun_data["version"] == first_data["version"]
        assert rerun_data["promoted"] is False
    finally:
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


# ============================================================================
# Scenario 5: Snapshot-backfill-on-cached-main branching (the new contract)
#
# These cover the new "if main exists, check snapshots; if any are
# missing, run snapshots-only" branching. Each scenario letter mirrors
# the plan -- ``A`` (all present) ... ``K`` (local_first + no repo +
# missing).
# ============================================================================


def _wait_for_terminal_status(
    client: TestClient, job_id: str, max_attempts: int = 5
) -> dict:
    """Poll ``GET /train/status/{job_id}`` until the background task
    leaves the ``in_progress`` / ``pending`` states."""
    for _ in range(max_attempts):
        resp = client.get(f"/train/status/{job_id}")
        if resp.status_code != 200:
            continue
        body = resp.json()
        if body["status"] not in {"pending", "in_progress"}:
            return body
    raise AssertionError(
        f"Job {job_id} did not reach a terminal status within {max_attempts} polls"
    )


def _seed_main_version(client: TestClient) -> str:
    """Drive the route once to fully populate the main version + all
    snapshots. Returns the cached version string for follow-up
    assertions."""
    body = _wait_for_terminal_response(client, "/train/lstm")
    return body["version"]


def test_train_lstm_cached_main_all_snapshots_present_returns_200(
    client_with_backfill_mocks,
):
    """Scenario A: cached main + all snapshots on disk -> 200 fast path."""
    _seed_main_version(client_with_backfill_mocks)

    response = client_with_backfill_mocks.post("/train/lstm", json={})
    assert response.status_code == 200, response.text
    data = response.json()
    assert "version" in data
    assert "metrics" in data


def test_train_lstm_cached_main_one_historical_missing_returns_202(
    client_with_backfill_mocks,
):
    """Scenario B: delete one historical snapshot -> 202 + snapshots-only job."""
    _seed_main_version(client_with_backfill_mocks)

    snapshot_storage = SnapshotLocalStorage("lstm_halal_new")
    snapshots = sorted(snapshot_storage.list_snapshots())
    # Drop the earliest historical snapshot from disk
    earliest = snapshots[0]
    for snap_dir in snapshot_storage.hashed_snapshot_dirs_for_cutoff(earliest):
        import shutil

        shutil.rmtree(snap_dir)

    response = client_with_backfill_mocks.post("/train/lstm", json={})
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]
    # Snapshots-only jobs are keyed under ``{bucket}_snapshots``
    assert job_id.startswith("lstm_halal_new_snapshots:")

    # Drive job to completion and assert the missing snapshot is back
    final = _wait_for_terminal_status(client_with_backfill_mocks, job_id)
    assert final["status"] == "completed", final
    snapshots_after = {
        s.isoformat() for s in SnapshotLocalStorage("lstm_halal_new").list_snapshots()
    }
    assert earliest.isoformat() in snapshots_after


def test_train_lstm_cached_main_end_window_missing_warn_and_skip(
    client_with_backfill_mocks, caplog
):
    """Scenario C: end-window snapshot missing while main is cached ->
    snapshots-only path warns and skips it (does not retrain main)."""
    import logging

    _seed_main_version(client_with_backfill_mocks)

    snapshot_storage = SnapshotLocalStorage("lstm_halal_new")
    snapshots = sorted(snapshot_storage.list_snapshots())
    end_window = snapshots[-1]
    for snap_dir in snapshot_storage.hashed_snapshot_dirs_for_cutoff(end_window):
        import shutil

        shutil.rmtree(snap_dir)

    caplog.set_level(logging.WARNING, logger="brain_api.routes.training.snapshot_phase")

    response = client_with_backfill_mocks.post("/train/lstm", json={})
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]
    final = _wait_for_terminal_status(client_with_backfill_mocks, job_id)
    assert final["status"] == "completed"

    # End-window snapshot must NOT have been recreated; warn must be emitted.
    snapshots_after = sorted(SnapshotLocalStorage("lstm_halal_new").list_snapshots())
    assert end_window not in snapshots_after, (
        "Snapshots-only path must not regenerate the end-window snapshot. "
        f"Found: {[s.isoformat() for s in snapshots_after]}"
    )
    warning_messages = [
        r.message for r in caplog.records if r.levelno >= logging.WARNING
    ]
    assert any("End-of-window snapshot" in m for m in warning_messages), (
        f"Expected warn-and-skip log, got: {warning_messages}"
    )


def test_train_lstm_cached_main_dedup_concurrent_snapshots_only_jobs(
    client_with_backfill_mocks,
):
    """Scenario D: a second POST while the snapshots-only job is in
    flight returns 202 with the same job id (no duplicate work)."""
    _seed_main_version(client_with_backfill_mocks)

    snapshot_storage = SnapshotLocalStorage("lstm_halal_new")
    snapshots = sorted(snapshot_storage.list_snapshots())
    for snap_dir in snapshot_storage.hashed_snapshot_dirs_for_cutoff(snapshots[0]):
        import shutil

        shutil.rmtree(snap_dir)

    response1 = client_with_backfill_mocks.post("/train/lstm", json={})
    assert response1.status_code == 202
    job_id1 = response1.json()["job_id"]

    final = _wait_for_terminal_status(client_with_backfill_mocks, job_id1)
    assert final["status"] == "completed"

    # Subsequent POST after backfill completes returns 200 fast path
    response2 = client_with_backfill_mocks.post("/train/lstm", json={})
    assert response2.status_code == 200


def test_train_lstm_cached_main_skip_snapshot_returns_200_fast(
    client_with_mocks,
):
    """Scenario G: ``?skip_snapshot=true`` bypasses the snapshot scan
    entirely and returns 200 even when no snapshots exist on disk."""
    response1 = client_with_mocks.post(TRAIN_URL_NO_SNAPSHOT, json={})
    assert response1.status_code == 202

    response2 = client_with_mocks.post(TRAIN_URL_NO_SNAPSHOT, json={})
    assert response2.status_code == 200, response2.text
    # No 202 path -- proves we never schedule a snapshots-only job
    # when the operator opted out of snapshot bookkeeping.


def test_train_lstm_cached_main_hf_first_no_repo_returns_503(
    client_with_backfill_mocks, monkeypatch
):
    """Scenario J: ``hf_first`` policy + the bucket has no HF repo
    configured -> 503 from the synchronous inventory scan."""
    _seed_main_version(client_with_backfill_mocks)

    snapshot_storage = SnapshotLocalStorage("lstm_halal_new")
    snapshots = sorted(snapshot_storage.list_snapshots())
    for snap_dir in snapshot_storage.hashed_snapshot_dirs_for_cutoff(snapshots[0]):
        import shutil

        shutil.rmtree(snap_dir)

    from brain_api.storage.policy import StoragePolicy

    # ``count_missing_snapshots`` reads the env-default via the rebound
    # name on its module, so we patch there (not on the storage policy
    # module itself).
    monkeypatch.setattr(
        "brain_api.core.forecaster_snapshot_identity.get_storage_policy",
        lambda: StoragePolicy.HF_FIRST,
    )

    response = client_with_backfill_mocks.post("/train/lstm", json={})
    assert response.status_code == 503, response.text
    assert "hf_first" in response.text


def test_train_lstm_cached_main_local_first_no_repo_with_missing_returns_202(
    client_with_backfill_mocks,
):
    """Scenario K: ``local_first`` + no HF repo + at least one missing
    -> 202 + snapshots-only job (the policy permits local-only)."""
    _seed_main_version(client_with_backfill_mocks)

    snapshot_storage = SnapshotLocalStorage("lstm_halal_new")
    snapshots = sorted(snapshot_storage.list_snapshots())
    for snap_dir in snapshot_storage.hashed_snapshot_dirs_for_cutoff(snapshots[0]):
        import shutil

        shutil.rmtree(snap_dir)

    response = client_with_backfill_mocks.post("/train/lstm", json={})
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]
    assert job_id.startswith("lstm_halal_new_snapshots:")
    final = _wait_for_terminal_status(client_with_backfill_mocks, job_id)
    assert final["status"] == "completed"

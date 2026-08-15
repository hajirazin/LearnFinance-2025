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
from brain_api.storage.forecaster_snapshots import SnapshotLocalStorage
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
    from datetime import date, timedelta

    n_samples = 10
    anchors = np.array(
        [date(2024, 1, 5) + timedelta(days=7 * i) for i in range(n_samples)],
        dtype=object,
    )
    return DatasetResult(
        X=np.random.randn(n_samples, config.context_length, 5),
        y=np.random.randn(n_samples, 5, 5),
        feature_scaler=StandardScaler(),
        anchor_dates=anchors,
    )


def _mock_trainer(
    X, y, feature_scaler, config, shutdown_event=None, anchor_dates=None
) -> TrainingResult:
    hf_config = config.to_hf_config()
    model = PatchTSTForPrediction(hf_config)
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


def _mock_trainer_worse(
    X, y, feature_scaler, config, shutdown_event=None, anchor_dates=None
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
        best_epoch=1,
        stopped_epoch=1,
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


def _patch_india_patchtst_backfill_internals(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the heavy compute helpers used by ``_backfill_patchtst_snapshots``.

    Backfill mechanics now live in
    :mod:`brain_api.routes.training.snapshot_phase`, shared between
    US and India PatchTST. Patching there covers both endpoints in
    one place.
    """
    from brain_api.routes.training import snapshot_phase

    monkeypatch.setattr(snapshot_phase, "patchtst_load_prices", _mock_price_loader)
    monkeypatch.setattr(
        snapshot_phase, "align_multivariate_data", lambda prices, config: prices
    )
    monkeypatch.setattr(
        snapshot_phase,
        "patchtst_build_dataset",
        lambda aligned_features, prices, config: _mock_dataset_builder(
            aligned_features, prices, config
        ),
    )
    monkeypatch.setattr(
        snapshot_phase,
        "patchtst_train_model",
        lambda X,
        y,
        feature_scaler,
        config,
        shutdown_event=None,
        anchor_dates=None: _mock_trainer(
            X, y, feature_scaler, config, shutdown_event=shutdown_event
        ),
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


@pytest.fixture
def client_india_with_backfill_mocks(temp_india_storage, monkeypatch):
    """India test client with backfill internals mocked too.

    Mirrors :func:`client_india` plus the route-module monkeypatches
    so the ``skip_snapshot=False`` branch runs in milliseconds while
    still exercising the real on-disk snapshot writes.
    """
    _override_india_patchtst_bucket(monkeypatch, temp_india_storage)
    app.dependency_overrides[get_patchtst_price_loader] = lambda: _mock_price_loader
    app.dependency_overrides[get_patchtst_dataset_builder] = (
        lambda: _mock_dataset_builder
    )
    app.dependency_overrides[get_patchtst_trainer] = lambda: _mock_trainer
    _patch_india_patchtst_backfill_internals(monkeypatch)

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


def test_train_patchtst_india_returns_required_fields(client_india, temp_india_storage):
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
    assert data["metrics"]["best_epoch"] == 1
    assert data["metrics"]["stopped_epoch"] == 1
    on_disk = temp_india_storage.read_metadata(data["version"])
    assert on_disk is not None
    assert on_disk["metrics"]["best_epoch"] == 1
    assert on_disk["metrics"]["stopped_epoch"] == 1


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


def test_train_patchtst_india_promotes_even_when_worse_than_prior(monkeypatch):
    """India PatchTST always-promote: a healthy model promotes regardless of prior val_loss.

    Same policy as US PatchTST: the prior-comparison gate was removed
    in favor of guardrails on the new artifact's own health, so a
    healthy run with worse-than-prior val_loss MUST still promote.
    """
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
            data = r2b.json()
            assert data["promoted"] is True
            assert storage.read_current_version() == data["version"]
            assert storage.read_current_version() != first_version
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


# ============================================================================
# skip_snapshot branch coverage
#
# The bulk of the file passes ``?skip_snapshot=true`` for speed; these
# two tests assert both branches against on-disk state under the
# autouse ``isolate_forecaster_snapshots`` tmp dir.
# ============================================================================


def _wait_for_terminal_response(
    client: TestClient, url: str, max_attempts: int = 5
) -> dict:
    last_response = None
    for _ in range(max_attempts):
        last_response = client.post(url)
        if last_response.status_code == 200:
            return last_response.json()
    raise AssertionError(
        f"Training did not converge to 200 within {max_attempts} POSTs. "
        f"Last status: {last_response.status_code if last_response else 'n/a'}"
    )


def test_train_patchtst_india_skip_snapshot_true_writes_no_snapshot(client_india):
    """``?skip_snapshot=true`` must NOT create any India snapshot on disk."""
    _ = _wait_for_terminal_response(client_india, TRAIN_INDIA_URL)

    snapshot_storage = SnapshotLocalStorage("patchtst_nifty_shariah_500")
    assert snapshot_storage.list_snapshots() == [], (
        "skip_snapshot=true must not write any India snapshots, but found: "
        f"{snapshot_storage.list_snapshots()}"
    )


def test_train_patchtst_india_skip_snapshot_false_writes_snapshots(
    client_india_with_backfill_mocks,
):
    """Default (``skip_snapshot=false``) writes the end-date snapshot AND backfills history.

    Uses :func:`client_india_with_backfill_mocks` so the heavy compute
    inside ``_backfill_patchtst_snapshots`` is replaced by deterministic
    stubs while the real on-disk write logic still runs.
    """
    _ = _wait_for_terminal_response(
        client_india_with_backfill_mocks, "/train/patchtst/india"
    )

    snapshot_storage = SnapshotLocalStorage("patchtst_nifty_shariah_500")
    snapshots = snapshot_storage.list_snapshots()
    assert snapshots, (
        "skip_snapshot=false must write at least one India snapshot, but "
        "on-disk snapshot list was empty."
    )
    end_date_iso = "2024-12-27"
    end_date_snapshots = [s for s in snapshots if s.isoformat() == end_date_iso]
    assert end_date_snapshots, (
        f"Expected end-date snapshot {end_date_iso} on disk, got: "
        f"{[s.isoformat() for s in snapshots]}"
    )
    assert len(snapshots) > 1, (
        "Backfill must populate historical India snapshots in addition to "
        f"the end-date one. Got only: {[s.isoformat() for s in snapshots]}"
    )


# ============================================================================
# Snapshot-backfill-on-cached-main branching for the India endpoint
# (delegates to ``handle_patchtst_existing_metadata`` shared with US)
# ============================================================================


def _wait_for_terminal_status(
    client: TestClient, job_id: str, max_attempts: int = 5
) -> dict:
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


def test_train_patchtst_india_cached_main_all_snapshots_present_returns_200(
    client_india_with_backfill_mocks,
):
    """Scenario A: cached India main + all snapshots present -> 200."""
    _ = _wait_for_terminal_response(
        client_india_with_backfill_mocks, "/train/patchtst/india"
    )

    response = client_india_with_backfill_mocks.post("/train/patchtst/india")
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["signals_used"] == ["ohlcv"]


def test_train_patchtst_india_cached_main_one_historical_missing_returns_202(
    client_india_with_backfill_mocks,
):
    """Scenario B: missing historical snapshot triggers snapshots-only job
    on the dedicated ``patchtst_nifty_shariah_500_snapshots`` key."""
    _ = _wait_for_terminal_response(
        client_india_with_backfill_mocks, "/train/patchtst/india"
    )

    snapshot_storage = SnapshotLocalStorage("patchtst_nifty_shariah_500")
    snapshots = sorted(snapshot_storage.list_snapshots())
    earliest = snapshots[0]
    for snap_dir in snapshot_storage.hashed_snapshot_dirs_for_cutoff(earliest):
        import shutil

        shutil.rmtree(snap_dir)

    response = client_india_with_backfill_mocks.post("/train/patchtst/india")
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]
    assert job_id.startswith("patchtst_nifty_shariah_500_snapshots:")

    final = _wait_for_terminal_status(client_india_with_backfill_mocks, job_id)
    assert final["status"] == "completed"


def test_train_patchtst_india_cached_main_recreated_snapshot_preserves_ns_suffix(
    client_india_with_backfill_mocks,
):
    """India invariant: recreated snapshots must keep ``.NS`` symbols
    in the on-disk metadata (AGENTS.md India universe rule)."""
    import json

    _ = _wait_for_terminal_response(
        client_india_with_backfill_mocks, "/train/patchtst/india"
    )

    snapshot_storage = SnapshotLocalStorage("patchtst_nifty_shariah_500")
    snapshots = sorted(snapshot_storage.list_snapshots())
    earliest = snapshots[0]
    for snap_dir in snapshot_storage.hashed_snapshot_dirs_for_cutoff(earliest):
        import shutil

        shutil.rmtree(snap_dir)

    response = client_india_with_backfill_mocks.post("/train/patchtst/india")
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    final = _wait_for_terminal_status(client_india_with_backfill_mocks, job_id)
    assert final["status"] == "completed"

    storage_after = SnapshotLocalStorage("patchtst_nifty_shariah_500")
    recreated_dirs = storage_after.hashed_snapshot_dirs_for_cutoff(earliest)
    assert recreated_dirs, "snapshot dir was not recreated"
    metadata_path = recreated_dirs[0] / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata_symbols = metadata["symbols"]
    assert all(s.endswith(".NS") for s in metadata_symbols), (
        f"India snapshot metadata symbols must keep .NS suffix, got: {metadata_symbols}"
    )


def test_train_patchtst_india_cached_main_skip_snapshot_returns_200(
    client_india,
):
    """Scenario G: ``?skip_snapshot=true`` bypasses scan."""
    response1 = client_india.post(TRAIN_INDIA_URL)
    assert response1.status_code == 202

    response2 = client_india.post(TRAIN_INDIA_URL)
    assert response2.status_code == 200, response2.text


def test_train_patchtst_india_cached_main_hf_first_no_repo_returns_503(
    client_india_with_backfill_mocks, monkeypatch
):
    """Scenario J: ``hf_first`` + no HF repo on India bucket -> 503."""
    _ = _wait_for_terminal_response(
        client_india_with_backfill_mocks, "/train/patchtst/india"
    )

    snapshot_storage = SnapshotLocalStorage("patchtst_nifty_shariah_500")
    snapshots = sorted(snapshot_storage.list_snapshots())
    for snap_dir in snapshot_storage.hashed_snapshot_dirs_for_cutoff(snapshots[0]):
        import shutil

        shutil.rmtree(snap_dir)

    from brain_api.storage.policy import StoragePolicy

    monkeypatch.setattr(
        "brain_api.core.forecaster_snapshot_identity.get_storage_policy",
        lambda: StoragePolicy.HF_FIRST,
    )

    response = client_india_with_backfill_mocks.post("/train/patchtst/india")
    assert response.status_code == 503, response.text
    assert "patchtst_nifty_shariah_500" in response.text

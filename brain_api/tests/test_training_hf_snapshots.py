"""HF training reuse contracts with external work stubbed."""

import os
import tempfile
from dataclasses import replace
from unittest.mock import patch

from fastapi.testclient import TestClient

from brain_api.core.model_buckets import ModelType, get_bucket
from brain_api.main import app
from tests.training_hf_cases import (
    _FAKE_METADATA,
    _bind_lstm_bucket,
)


def test_lstm_hf_main_present_with_missing_snapshots_returns_202(monkeypatch):
    """HF says main exists but the snapshot inventory finds work to do
    (no snapshots on disk and no HF repo configured for snapshots);
    the route must enqueue a snapshots-only job rather than return
    200 cached.

    The snapshot phase background task is replaced with a no-op spy
    so the test never invokes the real ``load_prices_yfinance`` /
    ``train_model_pytorch`` pair. The route's scheduling contract --
    which is what this test pins -- only depends on the inventory
    scan and the registry key shape, not on the actual training.
    """
    from brain_api.routes.training import (
        get_dataset_builder,
        get_price_loader,
        get_trainer,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        _bind_lstm_bucket(monkeypatch, tmpdir)

        # Tripwires: the cached-main path must not reach the main
        # trainer DI chain at all.
        def _fail_if_called(*_a, **_kw):
            raise AssertionError(
                "Main trainer must not run when snapshots-only path is taken"
            )

        app.dependency_overrides[get_price_loader] = lambda: _fail_if_called
        app.dependency_overrides[get_dataset_builder] = lambda: _fail_if_called
        app.dependency_overrides[get_trainer] = lambda: _fail_if_called

        os.environ["LSTM_TRAIN_LOOKBACK_YEARS"] = "10"
        os.environ["LSTM_TRAIN_WINDOW_END_DATE"] = "2025-01-01"

        try:
            with (
                patch(
                    "brain_api.routes.training.lstm.try_load_existing_train_metadata",
                    return_value=_FAKE_METADATA,
                ),
                # CRITICAL: stub the snapshot phase. The scheduled
                # background task runs synchronously after TestClient
                # returns; without this, it would call real
                # ``load_prices_yfinance`` / ``train_model_pytorch``.
                patch(
                    "brain_api.routes.training.lstm._run_lstm_snapshot_phase",
                    return_value=None,
                ) as snapshot_phase_spy,
            ):
                client = TestClient(app)
                # No ?skip_snapshot=true -- exercise the new branching.
                response = client.post("/train/lstm", json={})

            assert response.status_code == 202, response.text
            body = response.json()
            assert body["job_id"].startswith("lstm_halal_new_snapshots:")
            assert snapshot_phase_spy.call_count == 1
            kwargs = snapshot_phase_spy.call_args.kwargs
            assert kwargs["log_prefix"] == "[LSTM Snapshots-only]"
        finally:
            app.dependency_overrides.clear()
            os.environ.pop("LSTM_TRAIN_LOOKBACK_YEARS", None)
            os.environ.pop("LSTM_TRAIN_WINDOW_END_DATE", None)


def test_patchtst_us_hf_main_present_with_missing_snapshots_returns_202(monkeypatch):
    """PatchTST US: same contract as the LSTM test above. HF main hits
    but snapshots are missing -> 202 + snapshots-only job key.

    Stubs the PatchTST snapshot phase so the background task does NOT
    invoke the real weekly price loading / fitting
    pair (which would download yfinance data + train a real Transformer).
    """
    from brain_api.core import model_buckets
    from brain_api.core.patchtst.config import US_WEEKLY_CONFIG
    from brain_api.routes.training.dependencies import (
        get_patchtst_config,
        get_patchtst_price_loader,
        get_weekly_patchtst_trainer,
    )
    from brain_api.storage.patchtst.local import PatchTSTHalalNewModelStorage

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PatchTSTHalalNewModelStorage(base_path=tmpdir)
        original = get_bucket(ModelType.PATCHTST, "halal_new")
        patched = replace(
            original,
            local_storage_class=lambda: storage,
            symbols_resolver=lambda: ["AAPL", "MSFT"],
        )
        monkeypatch.setitem(
            model_buckets._BUCKETS, (ModelType.PATCHTST, "halal_new"), patched
        )

        app.dependency_overrides[get_patchtst_config] = lambda: US_WEEKLY_CONFIG

        def _fail_if_called(*_a, **_kw):
            raise AssertionError(
                "Main trainer must not run when snapshots-only path is taken"
            )

        app.dependency_overrides[get_patchtst_price_loader] = lambda: _fail_if_called
        app.dependency_overrides[get_weekly_patchtst_trainer] = lambda: _fail_if_called

        try:
            with (
                patch(
                    "brain_api.routes.training.patchtst.try_load_existing_train_metadata",
                    return_value=_FAKE_METADATA,
                ),
                # Stub the snapshot phase -- see the LSTM sibling test
                # above for why this is critical.
                patch(
                    "brain_api.services.patchtst_training.backfill_weekly_patchtst_snapshots",
                    return_value=__import__("datetime").date(2017, 12, 31),
                ) as snapshot_phase_spy,
            ):
                client = TestClient(app)
                response = client.post("/train/patchtst", json={})

            assert response.status_code == 202, response.text
            body = response.json()
            assert body["job_id"].startswith("patchtst_halal_new_snapshots:")
            assert snapshot_phase_spy.call_count == 1
            assert snapshot_phase_spy.call_args.args[-1] is _fail_if_called
            # US route's log_prefix is "[PatchTST]" -> snapshots-only
            # variant becomes "[PatchTST] Snapshots-only" (the route
            # appends, the runner forwards verbatim).
            assert (
                snapshot_phase_spy.call_args.kwargs["log_prefix"]
                == "[PatchTST] Snapshots-only"
            )
        finally:
            app.dependency_overrides.clear()


def test_lstm_hf_main_present_with_all_snapshots_present_returns_200(monkeypatch):
    """HF (or local) reports every cutoff present -> 200 cached, no
    snapshots-only job created. Pins that the inventory scan's
    ``is_empty`` branch wins over the snapshot-job branch."""
    from brain_api.core.forecaster_snapshot_identity import (
        MissingSnapshotInventory,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        _bind_lstm_bucket(monkeypatch, tmpdir)

        os.environ["LSTM_TRAIN_LOOKBACK_YEARS"] = "10"
        os.environ["LSTM_TRAIN_WINDOW_END_DATE"] = "2025-01-01"

        empty_inventory = MissingSnapshotInventory(historical_cutoffs=())

        try:
            with (
                patch(
                    "brain_api.routes.training.lstm.try_load_existing_train_metadata",
                    return_value=_FAKE_METADATA,
                ),
                patch(
                    "brain_api.routes.training.lstm.count_missing_snapshots",
                    return_value=empty_inventory,
                ),
                patch(
                    "brain_api.routes.training.lstm._run_lstm_snapshot_phase",
                    return_value=None,
                ) as snapshot_phase_spy,
            ):
                client = TestClient(app)
                response = client.post("/train/lstm", json={})

            assert response.status_code == 200, response.text
            data = response.json()
            assert data["promoted"] is True
            # Critical: no background snapshots-only job ran.
            assert snapshot_phase_spy.call_count == 0
        finally:
            os.environ.pop("LSTM_TRAIN_LOOKBACK_YEARS", None)
            os.environ.pop("LSTM_TRAIN_WINDOW_END_DATE", None)


def test_lstm_hf_main_present_with_one_hf_snapshot_missing_returns_202(monkeypatch):
    """HF reports exactly one historical snapshot missing -> 202 with
    a snapshots-only job whose message reflects ``1 cutoff(s) missing``."""
    from datetime import date

    from brain_api.core.forecaster_snapshot_identity import (
        MissingSnapshotInventory,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        _bind_lstm_bucket(monkeypatch, tmpdir)

        os.environ["LSTM_TRAIN_LOOKBACK_YEARS"] = "10"
        os.environ["LSTM_TRAIN_WINDOW_END_DATE"] = "2025-01-01"

        partial_inventory = MissingSnapshotInventory(
            historical_cutoffs=(date(2022, 12, 31),),
        )

        try:
            with (
                patch(
                    "brain_api.routes.training.lstm.try_load_existing_train_metadata",
                    return_value=_FAKE_METADATA,
                ),
                patch(
                    "brain_api.routes.training.lstm.count_missing_snapshots",
                    return_value=partial_inventory,
                ),
                patch(
                    "brain_api.routes.training.lstm._run_lstm_snapshot_phase",
                    return_value=None,
                ) as snapshot_phase_spy,
            ):
                client = TestClient(app)
                response = client.post("/train/lstm", json={})

            assert response.status_code == 202, response.text
            body = response.json()
            assert body["job_id"].startswith("lstm_halal_new_snapshots:")
            assert "1 cutoff(s) missing" in body["message"]
            assert snapshot_phase_spy.call_count == 1
        finally:
            os.environ.pop("LSTM_TRAIN_LOOKBACK_YEARS", None)
            os.environ.pop("LSTM_TRAIN_WINDOW_END_DATE", None)

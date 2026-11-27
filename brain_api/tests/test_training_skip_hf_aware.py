"""HF training reuse contracts with external work stubbed."""

import os
import tempfile
from dataclasses import replace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from brain_api.core.model_buckets import ModelType, get_bucket
from brain_api.core.sac.readiness import SACTrainingReadiness
from brain_api.main import app
from tests.training_hf_cases import (
    _FAKE_METADATA,
    _bind_lstm_bucket,
    _fail_if_live_yfinance,
    _fake_metadata_with,
)


def test_lstm_endpoint_hf_cold_start_short_circuits_via_helper(monkeypatch):
    """LSTM: HF-hit short-circuit returns 200 + cached metadata.

    Pins that ``/train/lstm`` routes through
    ``try_load_existing_train_metadata`` (NOT the legacy
    ``storage.version_exists`` directly) so the wiped-local-cache
    case cannot silently retrain.
    """
    from brain_api.storage.lstm.local import LSTMHalalNewModelStorage

    with tempfile.TemporaryDirectory() as tmpdir:
        from brain_api.core import model_buckets
        from brain_api.routes.training import (
            get_dataset_builder,
            get_price_loader,
            get_trainer,
        )

        storage = LSTMHalalNewModelStorage(base_path=tmpdir)
        original = get_bucket(ModelType.LSTM, "halal_new")
        patched = replace(
            original,
            local_storage_class=lambda: storage,
            symbols_resolver=lambda: ["AAPL", "MSFT"],
        )
        monkeypatch.setitem(
            model_buckets._BUCKETS, (ModelType.LSTM, "halal_new"), patched
        )

        # Trainer dep stays a tripwire: if the short-circuit fails
        # and the route enters training, this asserts loudly.
        def _fail_if_called(*_args, **_kwargs):
            raise AssertionError(
                "Trainer must not be invoked when HF short-circuit hits"
            )

        app.dependency_overrides[get_price_loader] = lambda: _fail_if_called
        app.dependency_overrides[get_dataset_builder] = lambda: _fail_if_called
        app.dependency_overrides[get_trainer] = lambda: _fail_if_called

        os.environ["LSTM_TRAIN_LOOKBACK_YEARS"] = "10"
        os.environ["LSTM_TRAIN_WINDOW_END_DATE"] = "2025-01-01"

        try:
            with patch(
                "brain_api.routes.training.lstm.try_load_existing_train_metadata",
                return_value=_FAKE_METADATA,
            ):
                client = TestClient(app)
                response = client.post("/train/lstm?skip_snapshot=true", json={})

            assert response.status_code == 200, response.text
            data = response.json()
            assert data["data_window_start"] == "2016-01-01"
            assert data["data_window_end"] == "2026-04-30"
            assert data["metrics"] == {"loss": 0.42}
            assert data["promoted"] is True
            assert data["prior_version"] == "v2026-04-24-prev"
            assert data["failure_reasons"] == []
        finally:
            app.dependency_overrides.clear()
            os.environ.pop("LSTM_TRAIN_LOOKBACK_YEARS", None)
            os.environ.pop("LSTM_TRAIN_WINDOW_END_DATE", None)


def test_patchtst_us_endpoint_hf_cold_start_short_circuits_via_helper(monkeypatch):
    """PatchTST US route: HF-hit short-circuit preserves the
    PatchTST-specific tail (``num_input_channels`` + ``signals_used``)."""
    from brain_api.core import model_buckets
    from brain_api.core.patchtst.config import US_WEEKLY_CONFIG
    from brain_api.routes.training.dependencies import get_patchtst_config
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

        # Override the config dep so the route uses a deterministic
        # close-only config -- otherwise the route would import
        # the production config which may drift over time.
        app.dependency_overrides[get_patchtst_config] = lambda: US_WEEKLY_CONFIG

        try:
            with patch(
                "brain_api.routes.training.patchtst.try_load_existing_train_metadata",
                return_value=_FAKE_METADATA,
            ):
                client = TestClient(app)
                response = client.post("/train/patchtst?skip_snapshot=true", json={})

            assert response.status_code == 200, response.text
            data = response.json()
            assert data["promoted"] is True
            assert data["num_input_channels"] == 7
            assert data["signals_used"] == US_WEEKLY_CONFIG.feature_names
        finally:
            app.dependency_overrides.clear()


def test_patchtst_india_endpoint_hf_cold_start_short_circuits_via_helper(monkeypatch):
    """PatchTST India route: same contract as the US route; uses an
    independent bucket / HF repo so the test must not touch the US one."""
    from brain_api.core import model_buckets
    from brain_api.core.patchtst.config import INDIA_WEEKLY_CONFIG
    from brain_api.storage.patchtst.local import PatchTSTNiftyShariah500ModelStorage

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PatchTSTNiftyShariah500ModelStorage(base_path=tmpdir)
        original = get_bucket(ModelType.PATCHTST, "nifty_shariah_500")
        patched = replace(
            original,
            local_storage_class=lambda: storage,
            symbols_resolver=lambda: ["RELIANCE.NS", "TCS.NS"],
        )
        monkeypatch.setitem(
            model_buckets._BUCKETS,
            (ModelType.PATCHTST, "nifty_shariah_500"),
            patched,
        )

        try:
            with patch(
                "brain_api.routes.training.patchtst_india.try_load_existing_train_metadata",
                return_value={
                    **_FAKE_METADATA,
                    "training_evidence": {
                        "latest_label_end": "2026-04-24",
                        "latest_label_symbols": 2,
                    },
                },
            ):
                client = TestClient(app)
                response = client.post(
                    "/train/patchtst/india?skip_snapshot=true",
                    json={"universe": "nifty_shariah_500"},
                )

            assert response.status_code == 200, response.text
            data = response.json()
            assert data["promoted"] is True
            assert data["num_input_channels"] == 6
            assert data["signals_used"] == INDIA_WEEKLY_CONFIG.feature_names
        finally:
            app.dependency_overrides.clear()


@pytest.mark.parametrize("universe", ["halal_filtered", "halal"])
def test_sac_full_endpoint_hf_cold_start_short_circuits_via_helper(
    monkeypatch, universe: str
):
    """SAC full: parametrised across both A/B universes.

    The two SAC buckets (``sac_halal_filtered`` and ``sac_halal``)
    have independent ``current`` pointers and HF repos. Both must
    short-circuit independently -- a regression that wired the helper
    only to one bucket would silently retrain the other.
    """
    from datetime import date

    from brain_api.core import model_buckets
    from brain_api.routes.training.sac import full as full_route
    from brain_api.storage.sac import (
        SACHalalFilteredModelStorage,
        SACHalalModelStorage,
    )

    monkeypatch.setattr(
        full_route,
        "resolve_sac_training_window",
        lambda: (date(2018, 1, 1), date(2026, 4, 30)),
    )

    storage_cls = (
        SACHalalFilteredModelStorage
        if universe == "halal_filtered"
        else SACHalalModelStorage
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = storage_cls(base_path=tmpdir)
        original = get_bucket(ModelType.SAC, universe)
        # halal_filtered pins to 15; halal is variable. Use 15 for
        # both so the route's config resizer doesn't have to do any
        # work -- the short-circuit happens before training anyway.
        patched = replace(
            original,
            local_storage_class=lambda: storage,
            symbols_resolver=lambda: [f"S{i}" for i in range(15)],
        )
        monkeypatch.setitem(model_buckets._BUCKETS, (ModelType.SAC, universe), patched)

        sac_metadata = _fake_metadata_with(symbols=[f"S{i}" for i in range(15)])

        try:
            with patch(
                "brain_api.routes.training.sac.full.try_load_existing_train_metadata",
                return_value=sac_metadata,
            ):
                client = TestClient(app)
                response = client.post("/train/sac/full", json={"universe": universe})

            assert response.status_code == 200, response.text
            data = response.json()
            assert data["promoted"] is True
            # SAC-specific tail must round-trip via the call site;
            # this is the AGENTS.md rule #2 anchor for SAC.
            assert data["symbols_used"] == [f"S{i}" for i in range(15)]
        finally:
            app.dependency_overrides.clear()


@pytest.mark.parametrize("universe", ["halal_filtered", "halal"])
def test_sac_full_endpoint_force_true_bypasses_hf_helper(monkeypatch, universe: str):
    """SAC full: force=True bypasses the HF short-circuit helper and returns 202.

    Starlette TestClient drains BackgroundTasks before ``client.post``
    returns, so without a stub the route would invoke real
    ``load_prices_yfinance`` / ``train_sac``. The scheduling contract
    this test pins -- force bypasses the HF helper and enqueues a job
    -- does not depend on the actual training run.
    """
    from datetime import date, datetime

    from brain_api.core import model_buckets
    from brain_api.routes.training.job_registry import TrainingJob
    from brain_api.routes.training.sac import full as full_route
    from brain_api.storage.sac import (
        SACHalalFilteredModelStorage,
        SACHalalModelStorage,
    )

    monkeypatch.setattr(
        full_route,
        "resolve_sac_training_window",
        lambda: (date(2018, 1, 1), date(2026, 4, 30)),
    )

    storage_cls = (
        SACHalalFilteredModelStorage
        if universe == "halal_filtered"
        else SACHalalModelStorage
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = storage_cls(base_path=tmpdir)
        original = get_bucket(ModelType.SAC, universe)
        patched = replace(
            original,
            local_storage_class=lambda: storage,
            symbols_resolver=lambda: [f"S{i}" for i in range(15)],
        )
        monkeypatch.setitem(model_buckets._BUCKETS, (ModelType.SAC, universe), patched)

        sac_metadata = _fake_metadata_with(symbols=[f"S{i}" for i in range(15)])

        mock_job = TrainingJob(
            job_id="test", model_type="sac", status="running", started_at=datetime.now()
        )

        try:
            with (
                patch(
                    "brain_api.routes.training.sac.full.try_load_existing_train_metadata",
                    return_value=sac_metadata,
                ),
                patch(
                    "brain_api.routes.training.sac.full.get_or_create_job",
                    return_value=(mock_job, True),
                ),
                patch(
                    "brain_api.routes.training.sac.full.assess_sac_training_readiness",
                    return_value=SACTrainingReadiness.from_issues(
                        universe=universe,
                        symbols=[f"S{i}" for i in range(15)],
                        missing=[],
                        errors=[],
                    ),
                ),
                patch(
                    "brain_api.routes.training.sac.full.load_prices_yfinance",
                    side_effect=_fail_if_live_yfinance,
                ),
                patch(
                    "brain_api.routes.training.sac.full._run_sac_full_training",
                    return_value=None,
                ) as training_spy,
            ):
                client = TestClient(app)
                response = client.post(
                    "/train/sac/full", json={"universe": universe, "force": True}
                )

            assert response.status_code == 202, response.text
            assert training_spy.call_count == 1
        finally:
            app.dependency_overrides.clear()


def test_lstm_hf_main_present_with_hf_first_no_repo_returns_202(monkeypatch):
    """``hf_first`` with no HF repo still backfills missing local snapshots."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _bind_lstm_bucket(monkeypatch, tmpdir)

        os.environ["LSTM_TRAIN_LOOKBACK_YEARS"] = "10"
        os.environ["LSTM_TRAIN_WINDOW_END_DATE"] = "2025-01-01"
        monkeypatch.setenv("STORAGE_BACKEND", "hf_first")

        try:
            with (
                patch(
                    "brain_api.routes.training.lstm.try_load_existing_train_metadata",
                    return_value=_FAKE_METADATA,
                ),
                patch(
                    "brain_api.routes.training.lstm._run_lstm_snapshot_phase",
                    return_value=None,
                ),
            ):
                client = TestClient(app)
                response = client.post("/train/lstm", json={})

            assert response.status_code == 202, response.text
            assert response.json()["job_id"].startswith("lstm_halal_new_snapshots:")
        finally:
            os.environ.pop("LSTM_TRAIN_LOOKBACK_YEARS", None)
            os.environ.pop("LSTM_TRAIN_WINDOW_END_DATE", None)

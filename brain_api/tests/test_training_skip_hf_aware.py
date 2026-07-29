"""HF-aware short-circuit tests for every ``/train/*`` endpoint.

The five training endpoints (LSTM, PatchTST US-route, PatchTST India,
SAC full, SAC finetune) all delegate the "version already trained?"
check to :func:`brain_api.storage.policy.try_load_existing_train_metadata`.
Per the plan, the existing per-endpoint idempotency tests already
cover the local-hit branch (they POST twice and the second call
hits the local short-circuit). This file pins the HF-hit branch,
which is the bug the plan was written to fix: under
``STORAGE_BACKEND=hf_first`` a freshly-deployed Pi (or wiped Mac)
must NOT silently retrain work that already exists on HF.

Each test patches ``try_load_existing_train_metadata`` at the route
module to return a fake metadata dict (simulating "HF has this
version"), calls the endpoint, and asserts:

1. The response is 200 (not 202) -- the route short-circuited before
   enqueueing a BackgroundTask.
2. The response carries the patched metadata's fields -- proves the
   ``build_common_train_response_kwargs`` mapping wired correctly.
3. The PatchTST routes also expose the model-specific extras
   (``num_input_channels`` + ``signals_used``) so the per-model
   tail of the response shape didn't drift.
4. The SAC routes also expose ``symbols_used``.

Per AGENTS.md rule #2, the tests assert the model-specific extras
at each call site (PatchTST channels/signals, SAC symbols) so the
DRY refactor cannot silently drop a model-specific field.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import replace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from brain_api.core.model_buckets import ModelType, get_bucket
from brain_api.main import app

_FAKE_METADATA: dict = {
    "data_window": {"start": "2016-01-01", "end": "2026-04-30"},
    "metrics": {"loss": 0.42},
    "promoted": True,
    "prior_version": "v2026-04-24-prev",
    "failure_reasons": [],
}


def _fake_metadata_with(**overrides) -> dict:
    """Return a copy of the canonical fake metadata with overrides applied.

    Tests that need extra fields (``symbols`` for SAC) layer them on
    top so they cannot accidentally mutate the shared baseline.
    """
    out = dict(_FAKE_METADATA)
    out.update(overrides)
    return out


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# PatchTST (US route handler)
# ---------------------------------------------------------------------------


def test_patchtst_us_endpoint_hf_cold_start_short_circuits_via_helper(monkeypatch):
    """PatchTST US route: HF-hit short-circuit preserves the
    PatchTST-specific tail (``num_input_channels`` + ``signals_used``)."""
    from brain_api.core import model_buckets
    from brain_api.core.patchtst import PatchTSTConfig
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
        # (n_channels=5) config -- otherwise the route would import
        # the production config which may drift over time.
        app.dependency_overrides[get_patchtst_config] = lambda: PatchTSTConfig()

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
            assert data["num_input_channels"] == 5
            assert data["signals_used"] == ["ohlcv"]
        finally:
            app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# PatchTST India
# ---------------------------------------------------------------------------


def test_patchtst_india_endpoint_hf_cold_start_short_circuits_via_helper(monkeypatch):
    """PatchTST India route: same contract as the US route; uses an
    independent bucket / HF repo so the test must not touch the US one."""
    from brain_api.core import model_buckets
    from brain_api.core.patchtst import PatchTSTConfig
    from brain_api.routes.training.dependencies import get_patchtst_config
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

        app.dependency_overrides[get_patchtst_config] = lambda: PatchTSTConfig()

        try:
            with patch(
                "brain_api.routes.training.patchtst_india.try_load_existing_train_metadata",
                return_value=_FAKE_METADATA,
            ):
                client = TestClient(app)
                response = client.post(
                    "/train/patchtst/india?skip_snapshot=true",
                    json={"universe": "nifty_shariah_500"},
                )

            assert response.status_code == 200, response.text
            data = response.json()
            assert data["promoted"] is True
            assert data["num_input_channels"] == 5
            assert data["signals_used"] == ["ohlcv"]
        finally:
            app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# SAC full
# ---------------------------------------------------------------------------


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
    from brain_api.core import model_buckets
    from brain_api.storage.sac import (
        SACHalalFilteredModelStorage,
        SACHalalModelStorage,
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
    """SAC full: force=True bypasses the HF short-circuit helper and returns 202."""
    from datetime import datetime

    from brain_api.core import model_buckets
    from brain_api.routes.training.job_registry import TrainingJob
    from brain_api.storage.sac import (
        SACHalalFilteredModelStorage,
        SACHalalModelStorage,
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
            ):
                client = TestClient(app)
                response = client.post(
                    "/train/sac/full", json={"universe": universe, "force": True}
                )

            assert response.status_code == 202, response.text
        finally:
            app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# SAC finetune
# ---------------------------------------------------------------------------


def test_sac_finetune_endpoint_hf_cold_start_short_circuits_via_helper(monkeypatch):
    """SAC finetune: short-circuit returns 200 + cached metadata.

    Finetune is hard-pinned to ``sac_halal_filtered`` per AGENTS.md
    known limitation. Even so, the HF-aware skip MUST still apply to
    its ``-ft``-suffixed versions so a wiped local cache doesn't
    re-finetune work that already exists on HF.

    Note: finetune's prior-model preview path (loading the previous
    actor for symbol order + config) would normally run before the
    skip check. We patch that out so the test stays focused on the
    short-circuit; a real finetune integration test belongs in
    ``test_sac.py``.
    """
    from types import SimpleNamespace

    from brain_api.core import model_buckets
    from brain_api.core.sac import DEFAULT_SAC_CONFIG
    from brain_api.routes.training.dependencies import get_sac_storage
    from brain_api.storage.sac import SACHalalFilteredModelStorage

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = SACHalalFilteredModelStorage(base_path=tmpdir)
        original = get_bucket(ModelType.SAC, "halal_filtered")
        patched = replace(
            original,
            local_storage_class=lambda: storage,
            symbols_resolver=lambda: [f"S{i}" for i in range(15)],
        )
        monkeypatch.setitem(
            model_buckets._BUCKETS, (ModelType.SAC, "halal_filtered"), patched
        )
        app.dependency_overrides[get_sac_storage] = lambda: storage

        sac_metadata = _fake_metadata_with(symbols=[f"S{i}" for i in range(15)])

        # Stub out the prior-metadata + prior-artifacts preview so
        # the test focuses exclusively on the
        # ``try_load_existing_train_metadata`` short-circuit.
        prior_artifacts = SimpleNamespace(
            symbol_order=[f"S{i}" for i in range(15)],
            config=DEFAULT_SAC_CONFIG,
        )

        try:
            with (
                patch(
                    "brain_api.routes.training.sac.finetune.get_prior_metadata_for_bucket",
                    return_value={"version": "v2026-04-24-prev"},
                ),
                patch(
                    "brain_api.storage.policy.load_current_artifacts_for_bucket",
                    return_value=prior_artifacts,
                ),
                patch(
                    "brain_api.routes.training.sac.finetune.try_load_existing_train_metadata",
                    return_value=sac_metadata,
                ),
            ):
                client = TestClient(app)
                response = client.post("/train/sac/finetune", json={})

            assert response.status_code == 200, response.text
            data = response.json()
            assert data["promoted"] is True
            assert data["symbols_used"] == [f"S{i}" for i in range(15)]
        finally:
            app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# HF-aware skip x snapshot inventory interaction matrix
#
# These tests pin the new contract: even when
# ``try_load_existing_train_metadata`` says "main is cached", the
# forecaster routes still consult the snapshot inventory before
# returning 200. The four rows below cover the cross product of
#
#   (HF main hits) x (snapshot inventory empty | non-empty | unable-to-scan)
#
# for both forecaster families. SAC routes do not have a snapshot
# inventory and stay covered by the existing 200-cached tests above.
# ---------------------------------------------------------------------------


def _bind_lstm_bucket(monkeypatch, tmpdir, *, symbols=None):
    """Helper: rebind the LSTM halal_new bucket to a tmpdir storage."""
    from brain_api.core import model_buckets
    from brain_api.storage.lstm.local import LSTMHalalNewModelStorage

    storage = LSTMHalalNewModelStorage(base_path=tmpdir)
    original = get_bucket(ModelType.LSTM, "halal_new")
    patched = replace(
        original,
        local_storage_class=lambda: storage,
        symbols_resolver=lambda: list(symbols or ["AAPL", "MSFT"]),
    )
    monkeypatch.setitem(
        model_buckets._BUCKETS,
        (ModelType.LSTM, "halal_new"),
        patched,
    )
    return storage


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
            # Snapshot phase ran exactly once with main_artifacts=None
            # (the cached-main contract: no end-window recreation).
            assert snapshot_phase_spy.call_count == 1
            kwargs = snapshot_phase_spy.call_args.kwargs
            assert kwargs["main_artifacts"] is None
            assert kwargs["log_prefix"] == "[LSTM Snapshots-only]"
        finally:
            app.dependency_overrides.clear()
            os.environ.pop("LSTM_TRAIN_LOOKBACK_YEARS", None)
            os.environ.pop("LSTM_TRAIN_WINDOW_END_DATE", None)


def test_lstm_hf_main_present_with_hf_first_no_repo_returns_503(monkeypatch):
    """Operator chose ``hf_first`` but the LSTM bucket has no HF repo
    configured (the conftest clears the bucket-keyed HF env vars).
    The synchronous inventory scan must surface ``StoragePolicyError``
    as a 503 -- per AGENTS.md rule #1 (no silent fallback to local).

    No background task is enqueued in this branch, so no snapshot
    phase mock is needed; the synchronous inventory scan itself
    raises before any ``add_task`` call.
    """
    from brain_api.storage.policy import StoragePolicy

    with tempfile.TemporaryDirectory() as tmpdir:
        _bind_lstm_bucket(monkeypatch, tmpdir)

        os.environ["LSTM_TRAIN_LOOKBACK_YEARS"] = "10"
        os.environ["LSTM_TRAIN_WINDOW_END_DATE"] = "2025-01-01"

        monkeypatch.setattr(
            "brain_api.core.forecaster_snapshot_identity.get_storage_policy",
            lambda: StoragePolicy.HF_FIRST,
        )

        try:
            with patch(
                "brain_api.routes.training.lstm.try_load_existing_train_metadata",
                return_value=_FAKE_METADATA,
            ):
                client = TestClient(app)
                response = client.post("/train/lstm", json={})

            assert response.status_code == 503, response.text
            assert "hf_first" in response.text
        finally:
            os.environ.pop("LSTM_TRAIN_LOOKBACK_YEARS", None)
            os.environ.pop("LSTM_TRAIN_WINDOW_END_DATE", None)


def test_patchtst_us_hf_main_present_with_missing_snapshots_returns_202(monkeypatch):
    """PatchTST US: same contract as the LSTM test above. HF main hits
    but snapshots are missing -> 202 + snapshots-only job key.

    Stubs the PatchTST snapshot phase so the background task does NOT
    invoke the real ``patchtst_load_prices`` / ``patchtst_train_model``
    pair (which would download yfinance data + train a real Transformer).
    """
    from brain_api.core import model_buckets
    from brain_api.core.patchtst import PatchTSTConfig
    from brain_api.routes.training.dependencies import (
        get_patchtst_config,
        get_patchtst_dataset_builder,
        get_patchtst_price_loader,
        get_patchtst_trainer,
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

        app.dependency_overrides[get_patchtst_config] = lambda: PatchTSTConfig()

        def _fail_if_called(*_a, **_kw):
            raise AssertionError(
                "Main trainer must not run when snapshots-only path is taken"
            )

        app.dependency_overrides[get_patchtst_price_loader] = lambda: _fail_if_called
        app.dependency_overrides[get_patchtst_dataset_builder] = lambda: _fail_if_called
        app.dependency_overrides[get_patchtst_trainer] = lambda: _fail_if_called

        try:
            with (
                patch(
                    "brain_api.routes.training.patchtst.try_load_existing_train_metadata",
                    return_value=_FAKE_METADATA,
                ),
                # Stub the snapshot phase -- see the LSTM sibling test
                # above for why this is critical.
                patch(
                    "brain_api.routes.training.patchtst._run_patchtst_snapshot_phase",
                    return_value=None,
                ) as snapshot_phase_spy,
            ):
                client = TestClient(app)
                response = client.post("/train/patchtst", json={})

            assert response.status_code == 202, response.text
            body = response.json()
            assert body["job_id"].startswith("patchtst_halal_new_snapshots:")
            assert snapshot_phase_spy.call_count == 1
            assert snapshot_phase_spy.call_args.kwargs["main_artifacts"] is None
            # US route's log_prefix is "[PatchTST]" -> snapshots-only
            # variant becomes "[PatchTST] Snapshots-only" (the route
            # appends, the runner forwards verbatim).
            assert (
                snapshot_phase_spy.call_args.kwargs["log_prefix"]
                == "[PatchTST] Snapshots-only"
            )
        finally:
            app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# HF-aware "all snapshots present" / "one HF snapshot missing" matrix.
#
# These rows exercise the inventory scan's policy-aware branch. The
# scan itself is covered in detail in
# ``tests/test_forecaster_snapshots_inventory.py`` (TestCountMissingSnapshots),
# so here we only stub ``count_missing_snapshots`` at the route to
# focus on how the route reacts to its result. This avoids the cost
# of constructing a fake HF Hub client just to verify the route's
# 200 vs 202 branching, which is exactly what these tests pin.
# ---------------------------------------------------------------------------


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

        empty_inventory = MissingSnapshotInventory(
            end_window_cutoff=None, historical_cutoffs=()
        )

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
            end_window_cutoff=None,
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

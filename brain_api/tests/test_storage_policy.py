"""Tests for the storage policy resolver.

The plan ``storage_policy_local_first_hf_first`` rules out legacy
``STORAGE_BACKEND=local`` / ``STORAGE_BACKEND=hf`` values entirely.
This module pins that contract: invalid values raise at boot, the
default is ``LOCAL_FIRST``, and only the two new enum values are
accepted.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from brain_api.storage.policy import (
    ENV_STORAGE_BACKEND,
    StoragePolicy,
    get_storage_policy,
)


class TestGetStoragePolicy:
    """``get_storage_policy()`` env-var contract."""

    def test_unset_defaults_to_local_first(self, monkeypatch):
        monkeypatch.delenv(ENV_STORAGE_BACKEND, raising=False)
        assert get_storage_policy() is StoragePolicy.LOCAL_FIRST

    def test_empty_string_defaults_to_local_first(self, monkeypatch):
        monkeypatch.setenv(ENV_STORAGE_BACKEND, "")
        assert get_storage_policy() is StoragePolicy.LOCAL_FIRST

    def test_whitespace_only_defaults_to_local_first(self, monkeypatch):
        monkeypatch.setenv(ENV_STORAGE_BACKEND, "   ")
        assert get_storage_policy() is StoragePolicy.LOCAL_FIRST

    def test_local_first_resolves(self, monkeypatch):
        monkeypatch.setenv(ENV_STORAGE_BACKEND, "local_first")
        assert get_storage_policy() is StoragePolicy.LOCAL_FIRST

    def test_hf_first_resolves(self, monkeypatch):
        monkeypatch.setenv(ENV_STORAGE_BACKEND, "hf_first")
        assert get_storage_policy() is StoragePolicy.HF_FIRST

    @pytest.mark.parametrize(
        "legacy_value",
        ["local", "hf", "huggingface"],
    )
    def test_legacy_values_raise(self, monkeypatch, legacy_value):
        monkeypatch.setenv(ENV_STORAGE_BACKEND, legacy_value)
        with pytest.raises(ValueError):
            get_storage_policy()

    def test_garbage_value_raises(self, monkeypatch):
        monkeypatch.setenv(ENV_STORAGE_BACKEND, "garbage_value")
        with pytest.raises(ValueError):
            get_storage_policy()

    def test_case_sensitive(self, monkeypatch):
        """``StoragePolicy`` values are lowercase; case must match exactly."""
        monkeypatch.setenv(ENV_STORAGE_BACKEND, "LOCAL_FIRST")
        with pytest.raises(ValueError):
            get_storage_policy()


class TestStartupValidation:
    """``brain_api.main`` calls ``get_storage_policy()`` at import time;
    importing a fresh module copy under an invalid env var must fail.

    Both tests below MUST restore the original ``brain_api.main`` to
    ``sys.modules`` after the reimport. Without that restoration the
    next test that does ``from brain_api.main import shutdown_event``
    (the LSTM training route does this lazily inside its background
    task, see ``brain_api/routes/training/lstm.py``) re-executes
    ``brain_api/main.py`` -- which calls ``load_dotenv()``, which
    repopulates ``HF_LSTM_HALAL_NEW_MODEL_REPO`` from ``.env`` because
    the autouse ``isolate_from_env`` fixture has popped it. The result
    is a real, unmocked HuggingFace upload of model weights inside an
    unrelated test (~6s of network I/O + side effects on the real HF
    repo). Restoring the original module keeps it cached in
    ``sys.modules`` so subsequent imports are no-ops.
    """

    def test_invalid_env_fails_app_boot(self, monkeypatch):
        import importlib
        import sys

        original = sys.modules.get("brain_api.main")
        monkeypatch.setenv(ENV_STORAGE_BACKEND, "totally_invalid")
        # Drop any cached module so the import-time validation runs again.
        sys.modules.pop("brain_api.main", None)
        try:
            with pytest.raises(ValueError):
                importlib.import_module("brain_api.main")
        finally:
            sys.modules.pop("brain_api.main", None)
            if original is not None:
                sys.modules["brain_api.main"] = original

    def test_valid_env_boots(self, monkeypatch):
        import importlib
        import sys

        original = sys.modules.get("brain_api.main")
        monkeypatch.setenv(ENV_STORAGE_BACKEND, "hf_first")
        sys.modules.pop("brain_api.main", None)
        try:
            module = importlib.import_module("brain_api.main")
            # Smoke-check the FastAPI app responds to a basic request to
            # confirm boot finished without surfacing the policy error.
            client = TestClient(module.app)
            response = client.get("/")
            assert response.status_code in (200, 404)  # any deterministic status
        finally:
            sys.modules.pop("brain_api.main", None)
            if original is not None:
                sys.modules["brain_api.main"] = original


class TestEnsureSnapshotForBucketContract:
    """Surface-level contract for the snapshot helper. Per-policy x
    per-bucket exercise lives in ``test_hf_storage.py`` and the
    forecaster snapshot integration tests.
    """

    def test_hf_first_without_repo_raises(self, monkeypatch, tmp_path):
        """``hf_first`` requires the bucket's HF repo env to be set."""
        from datetime import date

        from brain_api.storage.forecaster_snapshots import SnapshotLocalStorage
        from brain_api.storage.policy import (
            StoragePolicyError,
            ensure_snapshot_for_bucket,
        )

        # Clear all relevant HF repo envs so ``_get_hf_repo`` returns None.
        for env in (
            "HF_LSTM_HALAL_NEW_MODEL_REPO",
            "HF_PATCHTST_HALAL_NEW_MODEL_REPO",
            "HF_PATCHTST_NIFTY_SHARIAH_500_MODEL_REPO",
        ):
            monkeypatch.delenv(env, raising=False)

        snapshot = SnapshotLocalStorage("lstm_halal_new", base_path=tmp_path)
        with pytest.raises(StoragePolicyError):
            ensure_snapshot_for_bucket(
                snapshot_storage=snapshot,
                cutoff_date=date(2020, 12, 31),
                policy=StoragePolicy.HF_FIRST,
            )

    def test_local_first_short_circuits_when_local_present(self, tmp_path):
        from datetime import date
        from unittest.mock import MagicMock

        from sklearn.preprocessing import StandardScaler

        from brain_api.storage.forecaster_snapshots import SnapshotLocalStorage
        from brain_api.storage.policy import (
            ensure_snapshot_for_bucket,
        )

        snapshot = SnapshotLocalStorage("lstm_halal_new", base_path=tmp_path)
        cutoff = date(2020, 12, 31)
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {}
        mock_config = MagicMock()
        mock_config.to_dict.return_value = {}
        snapshot.write_snapshot(cutoff, mock_model, StandardScaler(), mock_config, {})

        # Should return True without touching HF (no env, no exception).
        assert (
            ensure_snapshot_for_bucket(
                snapshot_storage=snapshot,
                cutoff_date=cutoff,
                policy=StoragePolicy.LOCAL_FIRST,
            )
            is True
        )


class TestLoadCurrentArtifactsForBucketContract:
    """Cold-start (HF main missing) under ``hf_first`` -> 503 for inference."""

    def test_hf_first_cold_start_raises_503(self, monkeypatch):
        from unittest.mock import MagicMock, patch

        from brain_api.storage.policy import load_current_artifacts_for_bucket

        local_storage = MagicMock()
        local_storage.read_current_version.return_value = None
        hf_storage = MagicMock()
        hf_storage.token = None
        bucket = MagicMock()
        bucket.local_storage_class = lambda: local_storage
        bucket.hf_storage_class = lambda **kwargs: hf_storage
        bucket.hf_repo_getter = lambda: "user/repo"
        bucket.bucket_name = "test_bucket"

        with (
            patch(
                "brain_api.storage.policy._fetch_hf_main_metadata",
                return_value=None,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            load_current_artifacts_for_bucket(
                bucket=bucket,
                model_label="Test Model",
                policy=StoragePolicy.HF_FIRST,
            )

        assert exc_info.value.status_code == 503
        assert "cold-start" in exc_info.value.detail.lower()

    def test_local_first_no_local_no_hf_raises_503(self):
        from unittest.mock import MagicMock

        from brain_api.storage.policy import load_current_artifacts_for_bucket

        local_storage = MagicMock()
        local_storage.load_current_artifacts.side_effect = ValueError(
            "No current model"
        )
        bucket = MagicMock()
        bucket.local_storage_class = lambda: local_storage
        bucket.hf_storage_class = MagicMock()
        bucket.hf_repo_getter = lambda: None
        bucket.bucket_name = "test_bucket"

        with pytest.raises(HTTPException) as exc_info:
            load_current_artifacts_for_bucket(
                bucket=bucket,
                model_label="Test Model",
                policy=StoragePolicy.LOCAL_FIRST,
            )

        assert exc_info.value.status_code == 503

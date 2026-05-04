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
        from unittest.mock import MagicMock, patch

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

        pinned_digest = "aaaaaaaaaaaa"

        snapshot.write_snapshot(
            cutoff_date=cutoff,
            snapshot_digest=pinned_digest,
            model=mock_model,
            feature_scaler=StandardScaler(),
            config=mock_config,
            metadata={},
        )

        with patch(
            "brain_api.core.forecaster_snapshot_identity."
            "expected_dec31_walkforward_snapshot_hash",
            return_value=pinned_digest,
        ):
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


class TestLoadCurrentArtifactsForBucketColdStartStatusCode:
    """Pin the ``cold_start_status_code`` knob.

    The knob exists so ``/models/active-symbols`` can preserve its
    legacy 400 cold-start contract while ``/inference/{lstm,patchtst,sac}``
    keep the AGENTS.md "cold-start surfaces as 503" default. It MUST
    only downgrade the two genuine cold-start branches (no model
    anywhere); transient failures (HF unreachable, hf_first without a
    repo, HF download failed) MUST stay on 503 regardless so an
    operator can distinguish "needs training" from a recoverable
    outage.
    """

    def test_hf_first_cold_start_with_status_code_400_raises_400(self, monkeypatch):
        """hf_first + HF main missing + ``cold_start_status_code=400`` -> 400."""
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
                cold_start_status_code=400,
            )

        assert exc_info.value.status_code == 400
        # The cold-start detail still surfaces (operator-friendly).
        assert "cold-start" in exc_info.value.detail.lower()

    def test_local_first_cold_start_with_status_code_400_raises_400(self):
        """local_first + no local + no HF repo + ``cold_start_status_code=400`` -> 400."""
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
                cold_start_status_code=400,
            )

        assert exc_info.value.status_code == 400
        assert "no hf repo is configured" in exc_info.value.detail.lower()

    def test_hf_first_unreachable_stays_503_even_with_cold_start_400(self, monkeypatch):
        """Transient HF metadata fetch failure stays 503.

        ``cold_start_status_code=400`` only applies to the two
        cold-start branches. A ``StoragePolicyError`` from the HF
        metadata fetch is transient (network, auth, rate limit) and
        must NOT be downgraded to 400 -- the operator needs to know
        the model could still exist on HF and a retry might succeed.
        """
        from unittest.mock import MagicMock, patch

        from brain_api.storage.policy import (
            StoragePolicyError,
            load_current_artifacts_for_bucket,
        )

        local_storage = MagicMock()
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
                side_effect=StoragePolicyError("HF unreachable"),
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            load_current_artifacts_for_bucket(
                bucket=bucket,
                model_label="Test Model",
                policy=StoragePolicy.HF_FIRST,
                cold_start_status_code=400,
            )

        # Transient => 503 regardless of the cold-start knob.
        assert exc_info.value.status_code == 503

    def test_hf_first_no_repo_stays_503_even_with_cold_start_400(self):
        """hf_first without an HF repo is a CONFIG error, not cold-start.

        The bucket may have a perfectly good local model; the policy
        is just refusing to use it because hf_first was selected
        without a repo. This is operator-correctable (set the env
        var or switch to local_first), so it must stay 503 to be
        distinguishable from the "train one first" cold-start case.
        """
        from unittest.mock import MagicMock

        from brain_api.storage.policy import load_current_artifacts_for_bucket

        local_storage = MagicMock()
        bucket = MagicMock()
        bucket.local_storage_class = lambda: local_storage
        bucket.hf_storage_class = MagicMock()
        bucket.hf_repo_getter = lambda: None
        bucket.bucket_name = "test_bucket"

        with pytest.raises(HTTPException) as exc_info:
            load_current_artifacts_for_bucket(
                bucket=bucket,
                model_label="Test Model",
                policy=StoragePolicy.HF_FIRST,
                cold_start_status_code=400,
            )

        assert exc_info.value.status_code == 503
        assert "hf_first policy requires an HF repo" in exc_info.value.detail

    def test_local_first_hf_download_failure_stays_503_even_with_cold_start_400(self):
        """local empty + HF download throws -> 503 (transient).

        Same rationale as the hf_first transient case: the model may
        exist on HF; the failure is recoverable. Must not be
        downgraded to 400 because that would tell an operator
        "retrain" when the right action is "retry".
        """
        from unittest.mock import MagicMock

        from brain_api.storage.policy import load_current_artifacts_for_bucket

        local_storage = MagicMock()
        local_storage.load_current_artifacts.side_effect = ValueError(
            "No current model"
        )
        hf_storage = MagicMock()
        hf_storage.download_model.side_effect = RuntimeError("HF down")
        bucket = MagicMock()
        bucket.local_storage_class = lambda: local_storage
        bucket.hf_storage_class = lambda **kwargs: hf_storage
        bucket.hf_repo_getter = lambda: "user/repo"
        bucket.bucket_name = "test_bucket"

        with pytest.raises(HTTPException) as exc_info:
            load_current_artifacts_for_bucket(
                bucket=bucket,
                model_label="Test Model",
                policy=StoragePolicy.LOCAL_FIRST,
                cold_start_status_code=400,
            )

        assert exc_info.value.status_code == 503
        assert "HF" in exc_info.value.detail


class TestTryLoadExistingTrainMetadata:
    """HF-aware idempotency skip used by every ``/train/*`` endpoint.

    The helper short-circuits retraining when the deterministic
    ``v{end_date}-{hash}`` version already exists locally OR on HF
    (under ``hf_first``). It must:

    * always check local first (cheap path, byte-equivalent to legacy);
    * only consult HF under ``hf_first`` (no behaviour change for
      ``local_first`` -- the default);
    * use ``revision=<version>`` to fetch only ``metadata.json`` from
      the matching HF branch (each version is its own branch);
    * return ``None`` (not raise) on transient HF failure so the
      caller proceeds with training -- per AGENTS.md rule #1, the
      idempotency optimization must NOT wedge a worker on a flaky
      HF outage.
    """

    def _bucket_with_local(self, local_storage, hf_repo: str | None = "user/repo"):
        from unittest.mock import MagicMock

        hf_storage = MagicMock()
        hf_storage.token = "hf_test_token"
        bucket = MagicMock()
        bucket.local_storage_class = lambda: local_storage
        bucket.hf_storage_class = lambda **kwargs: hf_storage
        bucket.hf_repo_getter = lambda: hf_repo
        bucket.bucket_name = "test_bucket"
        return bucket, hf_storage

    def test_local_hit_returns_local_metadata(self):
        """Local has the version -> return its metadata; never touch HF."""
        from unittest.mock import MagicMock, patch

        from brain_api.storage.policy import try_load_existing_train_metadata

        local_storage = MagicMock()
        local_storage.version_exists.return_value = True
        local_storage.read_metadata.return_value = {"local": "yes", "version": "v1"}
        bucket, _ = self._bucket_with_local(local_storage)

        with patch("brain_api.storage.policy.hf_hub_download") as mock_dl:
            result = try_load_existing_train_metadata(
                bucket=bucket,
                version="v2026-05-01-aaa",
                local_storage=local_storage,
                policy=StoragePolicy.HF_FIRST,
            )

        assert result == {"local": "yes", "version": "v1"}
        # Local hit must NOT touch HF -- both for cost reasons (one
        # round-trip avoided per train call) and because this proves
        # the local path stays byte-equivalent to the legacy block.
        assert mock_dl.call_count == 0

    def test_hf_first_hit_returns_hf_metadata(self, tmp_path):
        """Local empty + HF has the revision -> download metadata.json."""
        import json
        from unittest.mock import MagicMock, patch

        from brain_api.storage.policy import try_load_existing_train_metadata

        local_storage = MagicMock()
        local_storage.version_exists.return_value = False
        bucket, _ = self._bucket_with_local(local_storage)

        # Stage a real metadata.json on disk so the helper's
        # ``open(path)`` path is exercised end-to-end.
        meta_path = tmp_path / "metadata.json"
        meta_payload = {
            "version": "v2026-05-01-aaa",
            "data_window": {"start": "2016-01-01", "end": "2026-04-30"},
            "metrics": {"loss": 0.42},
            "promoted": True,
        }
        meta_path.write_text(json.dumps(meta_payload))

        with patch(
            "brain_api.storage.policy.hf_hub_download", return_value=str(meta_path)
        ) as mock_dl:
            result = try_load_existing_train_metadata(
                bucket=bucket,
                version="v2026-05-01-aaa",
                local_storage=local_storage,
                policy=StoragePolicy.HF_FIRST,
            )

        assert result == meta_payload
        # Pin the revision-pinning contract: the helper MUST request
        # the matching HF branch (not main, not a sibling). Drift
        # here would silently start returning OTHER versions'
        # metadata as a "hit" for THIS version.
        kwargs = mock_dl.call_args.kwargs
        assert kwargs["revision"] == "v2026-05-01-aaa"
        assert kwargs["filename"] == "metadata.json"
        assert kwargs["repo_id"] == "user/repo"

    def test_hf_first_revision_not_found_returns_none(self):
        """Local empty + HF revision 404 -> None (genuine miss)."""
        from unittest.mock import MagicMock, patch

        from huggingface_hub.utils import RevisionNotFoundError

        from brain_api.storage.policy import try_load_existing_train_metadata

        local_storage = MagicMock()
        local_storage.version_exists.return_value = False
        bucket, _ = self._bucket_with_local(local_storage)

        with patch(
            "brain_api.storage.policy.hf_hub_download",
            side_effect=RevisionNotFoundError("not found"),
        ):
            result = try_load_existing_train_metadata(
                bucket=bucket,
                version="v2026-05-01-aaa",
                local_storage=local_storage,
                policy=StoragePolicy.HF_FIRST,
            )

        assert result is None

    def test_hf_first_transient_failure_returns_none(self):
        """Transient HF failure (network, auth) -> None, NEVER raise.

        The helper is an idempotency optimization, not a correctness
        gate. Raising on a transient HF outage would block training
        even though training is the correct recovery action.
        """
        from unittest.mock import MagicMock, patch

        from brain_api.storage.policy import try_load_existing_train_metadata

        local_storage = MagicMock()
        local_storage.version_exists.return_value = False
        bucket, _ = self._bucket_with_local(local_storage)

        with patch(
            "brain_api.storage.policy.hf_hub_download",
            side_effect=ConnectionError("HF down"),
        ):
            result = try_load_existing_train_metadata(
                bucket=bucket,
                version="v2026-05-01-aaa",
                local_storage=local_storage,
                policy=StoragePolicy.HF_FIRST,
            )

        assert result is None

    def test_local_first_skips_hf_check(self):
        """Under local_first, HF must NOT be consulted on local miss.

        Backward-compat anchor: existing local_first deployments
        retrain on local-cache-empty hosts today. Switching this to
        also consult HF would be a behavior change and is out of
        scope for this plan -- only ``hf_first`` opts in.
        """
        from unittest.mock import MagicMock, patch

        from brain_api.storage.policy import try_load_existing_train_metadata

        local_storage = MagicMock()
        local_storage.version_exists.return_value = False
        bucket, _ = self._bucket_with_local(local_storage)

        with patch("brain_api.storage.policy.hf_hub_download") as mock_dl:
            result = try_load_existing_train_metadata(
                bucket=bucket,
                version="v2026-05-01-aaa",
                local_storage=local_storage,
                policy=StoragePolicy.LOCAL_FIRST,
            )

        assert result is None
        assert mock_dl.call_count == 0

    def test_no_hf_repo_under_hf_first_returns_none(self):
        """hf_first + bucket has no HF repo configured -> None (not raise).

        The training caller is the right place to surface "no HF
        repo" as an error if it cares -- for the idempotency skip
        we just treat it as "can't check HF, proceed with training".
        """
        from unittest.mock import MagicMock, patch

        from brain_api.storage.policy import try_load_existing_train_metadata

        local_storage = MagicMock()
        local_storage.version_exists.return_value = False
        bucket, _ = self._bucket_with_local(local_storage, hf_repo=None)

        with patch("brain_api.storage.policy.hf_hub_download") as mock_dl:
            result = try_load_existing_train_metadata(
                bucket=bucket,
                version="v2026-05-01-aaa",
                local_storage=local_storage,
                policy=StoragePolicy.HF_FIRST,
            )

        assert result is None
        assert mock_dl.call_count == 0


class TestBuildCommonTrainResponseKwargs:
    """Pin the 7 common kwargs every ``*TrainResponse`` shares.

    Drift here silently breaks the cached-response path on every
    training endpoint, so the contract is locked in this dedicated
    class. New required fields on ``TrainResponse`` would surface as
    a missing key here.
    """

    def test_returns_seven_required_keys(self):
        from brain_api.storage.policy import build_common_train_response_kwargs

        metadata = {
            "data_window": {"start": "2016-01-01", "end": "2026-04-30"},
            "metrics": {"loss": 0.5},
            "promoted": True,
            "prior_version": "v2026-04-24-bbb",
            "failure_reasons": ["sac_cagr_below_floor"],
        }

        result = build_common_train_response_kwargs("v2026-05-01-aaa", metadata)

        assert set(result.keys()) == {
            "version",
            "data_window_start",
            "data_window_end",
            "metrics",
            "promoted",
            "prior_version",
            "failure_reasons",
        }
        assert result["version"] == "v2026-05-01-aaa"
        assert result["data_window_start"] == "2016-01-01"
        assert result["data_window_end"] == "2026-04-30"
        assert result["metrics"] == {"loss": 0.5}
        assert result["promoted"] is True
        assert result["prior_version"] == "v2026-04-24-bbb"
        assert result["failure_reasons"] == ["sac_cagr_below_floor"]

    def test_missing_failure_reasons_defaults_to_empty_list(self):
        """Pre-guardrail metadata files have no ``failure_reasons``.

        Old artifacts on disk / HF must continue to deserialize so
        an idempotent rerun on legacy versions doesn't 500.
        """
        from brain_api.storage.policy import build_common_train_response_kwargs

        metadata = {
            "data_window": {"start": "2016-01-01", "end": "2026-04-30"},
            "metrics": {},
            "promoted": False,
        }

        result = build_common_train_response_kwargs("v2026-05-01-aaa", metadata)

        assert result["failure_reasons"] == []
        assert result["prior_version"] is None

    def test_promoted_false_passes_through(self):
        """`promoted=False` in metadata round-trips into the response."""
        from brain_api.storage.policy import build_common_train_response_kwargs

        metadata = {
            "data_window": {"start": "2016-01-01", "end": "2026-04-30"},
            "metrics": {},
            "promoted": False,
            "failure_reasons": ["sac_cagr_below_floor"],
        }

        result = build_common_train_response_kwargs("v2026-05-01-aaa", metadata)

        assert result["promoted"] is False
        assert result["failure_reasons"] == ["sac_cagr_below_floor"]

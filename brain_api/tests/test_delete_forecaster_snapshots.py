"""Delete hashed forecaster snapshots on disk and on Hugging Face."""

from datetime import date
from types import SimpleNamespace

import pytest
from huggingface_hub.errors import HfHubHTTPError
from requests import Response

from brain_api.storage.forecaster_snapshots import SnapshotLocalStorage
from brain_api.storage.forecaster_snapshots.delete_snapshots import (
    ForecasterSnapshotDeleteError,
    HuggingFaceSnapshotRepoNotConfiguredError,
    SnapshotStorageTarget,
    delete_all_forecaster_snapshots,
)

_DIGEST_A = "aaaaaaaaaaaa"
_DIGEST_B = "bbbbbbbbbbbb"
_CUTOFF_A = date(2019, 12, 31)
_CUTOFF_B = date(2020, 12, 31)


def _seed_dir(path) -> None:
    path.mkdir(parents=True)
    (path / "metadata.json").write_text("{}")


def _hashed_dir(tmp_path, bucket: str, cutoff: date, digest: str):
    return tmp_path / "models" / bucket / f"snapshot-{cutoff.isoformat()}-{digest}"


def _hf_error(status_code: int) -> HfHubHTTPError:
    response = Response()
    response.status_code = status_code
    return HfHubHTTPError(f"status {status_code}", response=response)


def _storage(tmp_path) -> SnapshotLocalStorage:
    return SnapshotLocalStorage("lstm", base_path=tmp_path)


class TestDeleteLocalSnapshot:
    def test_delete_local_snapshot_removes_only_hashed_dir(self, tmp_path):
        storage = _storage(tmp_path)
        models = tmp_path / "models" / "lstm_halal_new"
        hashed = models / f"snapshot-{_CUTOFF_A.isoformat()}-{_DIGEST_A}"
        version = models / "v2026-01-09-aaaaaaaaaaaa"
        rejected = models / "rejected" / f"snapshot-{_CUTOFF_A.isoformat()}-{_DIGEST_A}"
        legacy = models / f"snapshot-{_CUTOFF_A.isoformat()}"
        for directory in (hashed, version, rejected, legacy):
            _seed_dir(directory)
        (models / "current").write_text("v2026-01-09-aaaaaaaaaaaa")

        assert storage.delete_local_snapshot(_CUTOFF_A, _DIGEST_A) is True

        assert not hashed.exists()
        assert (version / "metadata.json").exists()
        assert (models / "current").read_text() == "v2026-01-09-aaaaaaaaaaaa"
        assert (rejected / "metadata.json").exists()
        assert (legacy / "metadata.json").exists()

    def test_delete_local_snapshot_absent_returns_false(self, tmp_path):
        storage = _storage(tmp_path)
        assert storage.delete_local_snapshot(_CUTOFF_A, _DIGEST_A) is False

    def test_delete_local_snapshot_refuses_non_directory(self, tmp_path):
        storage = _storage(tmp_path)
        path = _hashed_dir(tmp_path, "lstm_halal_new", _CUTOFF_A, _DIGEST_A)
        path.parent.mkdir(parents=True)
        path.write_text("not a directory")

        with pytest.raises(ValueError, match="not a directory"):
            storage.delete_local_snapshot(_CUTOFF_A, _DIGEST_A)

        assert path.read_text() == "not a directory"


class _FakeHfApi:
    def __init__(self, token=None):
        self.token = token
        self.deleted: list[dict] = []
        self.list_error: Exception | None = None
        self.delete_error_for: dict[str, Exception] = {}
        self.branches: list[str] = []

    def list_repo_refs(self, repo_id, repo_type):
        if self.list_error is not None:
            raise self.list_error
        return SimpleNamespace(
            branches=[SimpleNamespace(name=name) for name in self.branches]
        )

    def delete_branch(self, repo_id, repo_type, branch):
        error = self.delete_error_for.get(branch)
        self.deleted.append(
            {"repo_id": repo_id, "repo_type": repo_type, "branch": branch}
        )
        if error is not None:
            raise error


def _install_hf_api(monkeypatch, api: _FakeHfApi, repo: str | None) -> None:
    monkeypatch.setattr(
        "brain_api.storage.forecaster_snapshots.snapshot_hf.HfApi",
        lambda token=None: api,
    )
    monkeypatch.setattr(
        "brain_api.storage.forecaster_snapshots.local.get_hf_lstm_halal_new_model_repo",
        lambda: repo,
    )


class TestDeleteHfSnapshot:
    def test_list_hf_strict_raises_when_repo_missing(self, tmp_path, monkeypatch):
        _install_hf_api(monkeypatch, _FakeHfApi(), None)
        storage = _storage(tmp_path)

        with pytest.raises(HuggingFaceSnapshotRepoNotConfiguredError):
            storage.list_hf_snapshot_identities_strict()

    def test_list_hf_strict_propagates_api_error(self, tmp_path, monkeypatch):
        api = _FakeHfApi()
        api.list_error = RuntimeError("hub down")
        _install_hf_api(monkeypatch, api, "org/lstm")
        storage = _storage(tmp_path)

        with pytest.raises(RuntimeError, match="hub down"):
            storage.list_hf_snapshot_identities_strict()

    def test_delete_hf_snapshot_calls_delete_branch(self, tmp_path, monkeypatch):
        api = _FakeHfApi()
        _install_hf_api(monkeypatch, api, "org/lstm")
        storage = _storage(tmp_path)

        assert storage.delete_hf_snapshot(_CUTOFF_A, _DIGEST_A) is True
        assert api.deleted == [
            {
                "repo_id": "org/lstm",
                "repo_type": "model",
                "branch": f"snapshot-{_CUTOFF_A.isoformat()}-{_DIGEST_A}",
            }
        ]

    def test_delete_hf_snapshot_404_returns_false(self, tmp_path, monkeypatch):
        api = _FakeHfApi()
        branch = f"snapshot-{_CUTOFF_A.isoformat()}-{_DIGEST_A}"
        api.delete_error_for[branch] = _hf_error(404)
        _install_hf_api(monkeypatch, api, "org/lstm")
        storage = _storage(tmp_path)

        assert storage.delete_hf_snapshot(_CUTOFF_A, _DIGEST_A) is False

    def test_delete_hf_snapshot_other_error_propagates(self, tmp_path, monkeypatch):
        api = _FakeHfApi()
        branch = f"snapshot-{_CUTOFF_A.isoformat()}-{_DIGEST_A}"
        api.delete_error_for[branch] = _hf_error(500)
        _install_hf_api(monkeypatch, api, "org/lstm")
        storage = _storage(tmp_path)

        with pytest.raises(HfHubHTTPError):
            storage.delete_hf_snapshot(_CUTOFF_A, _DIGEST_A)


class TestDeleteAllForecasterSnapshots:
    def test_delete_all_local_leaves_hf_uncontacted(self, tmp_path, monkeypatch):
        hashed = _hashed_dir(tmp_path, "lstm_halal_new", _CUTOFF_A, _DIGEST_A)
        _seed_dir(hashed)
        storage = _storage(tmp_path)

        def _forbidden(*_args, **_kwargs):
            raise AssertionError("HF must not be contacted")

        monkeypatch.setattr(
            "brain_api.storage.forecaster_snapshots.snapshot_hf.HfApi",
            _forbidden,
        )
        storage.list_hf_snapshot_identities_strict = _forbidden
        storage.delete_hf_snapshot = _forbidden

        result = delete_all_forecaster_snapshots(storage, SnapshotStorageTarget.LOCAL)

        assert not hashed.exists()
        assert result.forecaster_bucket == "lstm_halal_new"
        assert result.storage is SnapshotStorageTarget.LOCAL
        assert len(result.deleted) == 1
        assert result.deleted[0].deleted_local is True
        assert result.deleted[0].deleted_hf is False

    def test_delete_all_hf_leaves_local_dir(self, tmp_path, monkeypatch):
        hashed = _hashed_dir(tmp_path, "lstm_halal_new", _CUTOFF_A, _DIGEST_A)
        _seed_dir(hashed)
        api = _FakeHfApi()
        api.branches = [f"snapshot-{_CUTOFF_A.isoformat()}-{_DIGEST_A}", "main"]
        _install_hf_api(monkeypatch, api, "org/lstm")
        storage = _storage(tmp_path)

        result = delete_all_forecaster_snapshots(storage, SnapshotStorageTarget.HF)

        assert hashed.exists()
        assert (
            api.deleted[0]["branch"] == f"snapshot-{_CUTOFF_A.isoformat()}-{_DIGEST_A}"
        )
        assert result.deleted[0].deleted_local is False
        assert result.deleted[0].deleted_hf is True

    def test_delete_all_both(self, tmp_path, monkeypatch):
        local_only = _hashed_dir(tmp_path, "lstm_halal_new", _CUTOFF_A, _DIGEST_A)
        _seed_dir(local_only)
        branch_a = f"snapshot-{_CUTOFF_A.isoformat()}-{_DIGEST_A}"
        branch_b = f"snapshot-{_CUTOFF_B.isoformat()}-{_DIGEST_B}"
        api = _FakeHfApi()
        api.branches = [branch_b]
        api.delete_error_for[branch_a] = _hf_error(404)
        _install_hf_api(monkeypatch, api, "org/lstm")
        storage = _storage(tmp_path)

        result = delete_all_forecaster_snapshots(storage, SnapshotStorageTarget.BOTH)

        assert not local_only.exists()
        deleted = {
            (item.cutoff_date, item.snapshot_digest): item for item in result.deleted
        }
        assert deleted[(_CUTOFF_A, _DIGEST_A)].deleted_local is True
        assert deleted[(_CUTOFF_A, _DIGEST_A)].deleted_hf is False
        assert deleted[(_CUTOFF_B, _DIGEST_B)].deleted_local is False
        assert deleted[(_CUTOFF_B, _DIGEST_B)].deleted_hf is True
        assert [call["branch"] for call in api.deleted] == [branch_a, branch_b]

    def test_delete_all_hf_without_repo_deletes_nothing(self, tmp_path, monkeypatch):
        hashed = _hashed_dir(tmp_path, "lstm_halal_new", _CUTOFF_A, _DIGEST_A)
        _seed_dir(hashed)
        _install_hf_api(monkeypatch, _FakeHfApi(), None)
        storage = _storage(tmp_path)

        with pytest.raises(HuggingFaceSnapshotRepoNotConfiguredError):
            delete_all_forecaster_snapshots(storage, SnapshotStorageTarget.BOTH)

        assert hashed.exists()

    def test_delete_all_empty_is_empty_result(self, tmp_path):
        storage = _storage(tmp_path)
        result = delete_all_forecaster_snapshots(storage, SnapshotStorageTarget.LOCAL)
        assert result.deleted == ()

    def test_partial_hf_failure_keeps_successful_deletes_and_raises(
        self, tmp_path, monkeypatch
    ):
        api = _FakeHfApi()
        branch_a = f"snapshot-{_CUTOFF_A.isoformat()}-{_DIGEST_A}"
        branch_b = f"snapshot-{_CUTOFF_B.isoformat()}-{_DIGEST_B}"
        api.branches = [branch_a, branch_b]
        api.delete_error_for[branch_b] = _hf_error(500)
        _install_hf_api(monkeypatch, api, "org/lstm")
        storage = _storage(tmp_path)

        with pytest.raises(ForecasterSnapshotDeleteError) as caught:
            delete_all_forecaster_snapshots(storage, SnapshotStorageTarget.HF)

        assert [call["branch"] for call in api.deleted] == [branch_a, branch_b]
        assert caught.value.deleted[0].snapshot_digest == _DIGEST_A
        assert caught.value.deleted[0].deleted_hf is True
        assert caught.value.failures[0].snapshot_digest == _DIGEST_B
        assert caught.value.failures[0].backend == "hf"

"""API tests for DELETE hashed forecaster snapshots."""

from datetime import date

import pytest
from fastapi.testclient import TestClient
from huggingface_hub.errors import HfHubHTTPError
from requests import Response

from brain_api.main import app
from brain_api.storage.forecaster_snapshots.local import (
    SnapshotLocalStorage as RealSnapshotLocalStorage,
)

_DIGEST = "aaaaaaaaaaaa"
_CUTOFF = date(2019, 12, 31)
_LSTM_URL = "/train/lstm/snapshots"
_US_URL = "/train/patchtst/snapshots"
_INDIA_URL = "/train/patchtst/india/snapshots"


def _hashed_dir(root, bucket: str, cutoff: date = _CUTOFF, digest: str = _DIGEST):
    return root / "models" / bucket / f"snapshot-{cutoff.isoformat()}-{digest}"


def _seed(root, bucket: str, cutoff: date = _CUTOFF, digest: str = _DIGEST):
    path = _hashed_dir(root, bucket, cutoff, digest)
    path.mkdir(parents=True)
    (path / "metadata.json").write_text("{}")
    return path


@pytest.fixture
def snapshot_root(tmp_path, monkeypatch):
    from brain_api.routes.training import delete_snapshots as routes

    def factory(forecaster_type, base_path=None, hf_token=None):
        return RealSnapshotLocalStorage(
            forecaster_type, base_path=tmp_path, hf_token=hf_token
        )

    monkeypatch.setattr(routes, "SnapshotLocalStorage", factory)
    return tmp_path


@pytest.fixture
def client():
    return TestClient(app)


def test_missing_universe_is_422(client, snapshot_root):
    response = client.delete(_LSTM_URL, params={"storage": "local"})
    assert response.status_code == 422


def test_missing_storage_is_422(client, snapshot_root):
    response = client.delete(_LSTM_URL, params={"universe": "halal_new"})
    assert response.status_code == 422


def test_invalid_storage_is_422(client, snapshot_root):
    response = client.delete(
        _LSTM_URL, params={"universe": "halal_new", "storage": "s3"}
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    ("url", "universe"),
    [
        (_LSTM_URL, "nifty_shariah_500"),
        (_US_URL, "nifty_shariah_500"),
        (_INDIA_URL, "halal_new"),
    ],
)
def test_universe_outside_allowlist_deletes_nothing(
    client, snapshot_root, url, universe
):
    lstm = _seed(snapshot_root, "lstm_halal_new")
    us = _seed(snapshot_root, "patchtst_halal_new")
    india = _seed(snapshot_root, "patchtst_nifty_shariah_500")

    response = client.delete(url, params={"universe": universe, "storage": "local"})

    assert response.status_code == 422
    assert lstm.exists()
    assert us.exists()
    assert india.exists()


def test_local_delete_is_bucket_isolated(client, snapshot_root, monkeypatch):
    def _forbidden(*_args, **_kwargs):
        raise AssertionError("HF must not be contacted")

    monkeypatch.setattr(
        "brain_api.storage.forecaster_snapshots.snapshot_hf.HfApi",
        _forbidden,
    )
    lstm = _seed(snapshot_root, "lstm_halal_new")
    us = _seed(snapshot_root, "patchtst_halal_new")
    current = snapshot_root / "models" / "lstm_halal_new" / "current"
    current.write_text("v2026-01-09-aaaaaaaaaaaa")

    response = client.delete(
        _LSTM_URL, params={"universe": "halal_new", "storage": "local"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["universe"] == "halal_new"
    assert body["forecaster_bucket"] == "lstm_halal_new"
    assert body["storage"] == "local"
    assert body["deleted"][0]["deleted_local"] is True
    assert body["deleted"][0]["deleted_hf"] is False
    assert not lstm.exists()
    assert us.exists()
    assert current.read_text() == "v2026-01-09-aaaaaaaaaaaa"


def test_us_and_india_patchtst_delete_only_their_bucket(client, snapshot_root):
    us = _seed(snapshot_root, "patchtst_halal_new")
    india = _seed(snapshot_root, "patchtst_nifty_shariah_500")

    us_response = client.delete(
        _US_URL, params={"universe": "halal_new", "storage": "local"}
    )
    assert us_response.status_code == 200
    assert us_response.json()["forecaster_bucket"] == "patchtst_halal_new"
    assert not us.exists()
    assert india.exists()

    india_response = client.delete(
        _INDIA_URL,
        params={"universe": "nifty_shariah_500", "storage": "local"},
    )
    assert india_response.status_code == 200
    assert india_response.json()["forecaster_bucket"] == "patchtst_nifty_shariah_500"
    assert not india.exists()


def test_hf_deletes_branch_and_leaves_local_dir(client, snapshot_root, monkeypatch):
    local_dir = _seed(snapshot_root, "lstm_halal_new")
    branch = f"snapshot-{_CUTOFF.isoformat()}-{_DIGEST}"
    deleted: list[dict] = []

    class _Api:
        def __init__(self, token=None):
            self.token = token

        def list_repo_refs(self, repo_id, repo_type):
            from types import SimpleNamespace

            return SimpleNamespace(branches=[SimpleNamespace(name=branch)])

        def delete_branch(self, repo_id, repo_type, branch):
            deleted.append(
                {"repo_id": repo_id, "repo_type": repo_type, "branch": branch}
            )

    monkeypatch.setattr(
        "brain_api.storage.forecaster_snapshots.snapshot_hf.HfApi",
        _Api,
    )
    monkeypatch.setattr(
        "brain_api.storage.forecaster_snapshots.local.get_hf_lstm_halal_new_model_repo",
        lambda: "org/lstm",
    )

    response = client.delete(
        _LSTM_URL, params={"universe": "halal_new", "storage": "hf"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["storage"] == "hf"
    assert body["deleted"][0]["deleted_local"] is False
    assert body["deleted"][0]["deleted_hf"] is True
    assert local_dir.exists()
    assert deleted == [{"repo_id": "org/lstm", "repo_type": "model", "branch": branch}]


def test_both_without_repo_is_422_and_leaves_local_dir(
    client, snapshot_root, monkeypatch
):
    local_dir = _seed(snapshot_root, "lstm_halal_new")
    monkeypatch.setattr(
        "brain_api.storage.forecaster_snapshots.local.get_hf_lstm_halal_new_model_repo",
        lambda: None,
    )

    response = client.delete(
        _LSTM_URL, params={"universe": "halal_new", "storage": "both"}
    )

    assert response.status_code == 422
    assert local_dir.exists()


def test_empty_bucket_returns_empty_deleted(client, snapshot_root):
    response = client.delete(
        _LSTM_URL, params={"universe": "halal_new", "storage": "local"}
    )
    assert response.status_code == 200
    assert response.json()["deleted"] == []


def test_hf_list_failure_is_502_and_deletes_nothing(client, snapshot_root, monkeypatch):
    local_dir = _seed(snapshot_root, "lstm_halal_new")

    class _Api:
        def __init__(self, token=None):
            pass

        def list_repo_refs(self, repo_id, repo_type):
            raise RuntimeError("hub down")

    monkeypatch.setattr(
        "brain_api.storage.forecaster_snapshots.snapshot_hf.HfApi",
        _Api,
    )
    monkeypatch.setattr(
        "brain_api.storage.forecaster_snapshots.local.get_hf_lstm_halal_new_model_repo",
        lambda: "org/lstm",
    )

    response = client.delete(
        _LSTM_URL, params={"universe": "halal_new", "storage": "hf"}
    )

    assert response.status_code == 502
    assert "hub down" in response.json()["detail"]
    assert local_dir.exists()


def test_partial_hf_failure_is_500(client, snapshot_root, monkeypatch):
    branch_a = f"snapshot-{_CUTOFF.isoformat()}-{_DIGEST}"
    branch_b = "snapshot-2020-12-31-bbbbbbbbbbbb"

    class _Api:
        def __init__(self, token=None):
            pass

        def list_repo_refs(self, repo_id, repo_type):
            from types import SimpleNamespace

            return SimpleNamespace(
                branches=[
                    SimpleNamespace(name=branch_a),
                    SimpleNamespace(name=branch_b),
                ]
            )

        def delete_branch(self, repo_id, repo_type, branch):
            if branch == branch_b:
                response = Response()
                response.status_code = 500
                raise HfHubHTTPError("nope", response=response)

    monkeypatch.setattr(
        "brain_api.storage.forecaster_snapshots.snapshot_hf.HfApi",
        _Api,
    )
    monkeypatch.setattr(
        "brain_api.storage.forecaster_snapshots.local.get_hf_lstm_halal_new_model_repo",
        lambda: "org/lstm",
    )

    response = client.delete(
        _LSTM_URL, params={"universe": "halal_new", "storage": "hf"}
    )

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail["message"] == "Failed to delete one or more forecaster snapshots"
    assert detail["deleted"][0]["snapshot_digest"] == _DIGEST
    assert detail["failures"][0]["snapshot_digest"] == "bbbbbbbbbbbb"
    assert detail["failures"][0]["backend"] == "hf"

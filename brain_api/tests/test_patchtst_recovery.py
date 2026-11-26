"""Publication failures reuse the completed weekly fit."""

from dataclasses import replace
from datetime import date
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from brain_api.core import model_buckets
from brain_api.core.model_buckets import ModelType, get_bucket
from brain_api.core.patchtst.config import INDIA_WEEKLY_CONFIG, create_patchtst_model
from brain_api.core.patchtst.version import compute_version
from brain_api.core.patchtst.weekly_training import PatchTSTTargetScale
from brain_api.main import app
from brain_api.routes.training.dependencies import get_india_patchtst_config
from brain_api.storage.metadata import create_training_metadata
from brain_api.storage.patchtst.local import PatchTSTNiftyShariah500ModelStorage


@pytest.mark.parametrize("publication_pending", [True, False])
def test_retry_failed_hf_publication_reuses_saved_weights(
    monkeypatch, tmp_path, publication_pending
):
    config = INDIA_WEEKLY_CONFIG
    start, end = date(2024, 1, 1), date(2026, 11, 20)
    symbols = [f"STOCK{i}.NS" for i in range(10)]
    version = compute_version(start, end, symbols, config)
    storage = PatchTSTNiftyShariah500ModelStorage(tmp_path)
    model = create_patchtst_model(config)
    metadata = create_training_metadata(
        model_type=storage.model_type,
        version=version,
        data_window_start=str(start),
        data_window_end=str(end),
        symbols=symbols,
        config_dict=config.to_dict(),
        train_loss=0.01,
        val_loss=0.02,
        baseline_loss=0.03,
        best_epoch=1,
        stopped_epoch=9,
        promoted=True,
        prior_version=None,
    )
    metadata.update(
        publication_pending=publication_pending,
        training_evidence={"latest_label_end": str(end), "latest_label_symbols": 10},
    )
    storage.write_artifacts(version, model, PatchTSTTargetScale(0.03), config, metadata)
    upload = Mock(
        side_effect=[ConnectionError("Hub down"), SimpleNamespace(repo_id="test/india")]
    )
    hf = SimpleNamespace(upload_model=upload)
    original = get_bucket(ModelType.PATCHTST, "nifty_shariah_500")
    bucket = replace(
        original,
        local_storage_class=lambda: storage,
        symbols_resolver=lambda: symbols,
        hf_repo_getter=lambda: "test/india",
        hf_storage_class=lambda **kwargs: hf,
    )
    monkeypatch.setitem(
        model_buckets._BUCKETS, (ModelType.PATCHTST, bucket.universe), bucket
    )
    monkeypatch.setattr(
        "brain_api.routes.training.patchtst_india.resolve_training_window",
        lambda: (start, end),
    )
    forbidden_fit = Mock(
        side_effect=AssertionError("Saved checkpoint must not be refit")
    )
    monkeypatch.setattr(
        "brain_api.routes.training.patchtst.fit_weekly_patchtst", forbidden_fit
    )
    monkeypatch.setitem(
        app.dependency_overrides, get_india_patchtst_config, lambda: config
    )
    client = TestClient(app)
    for expected in ("failed", "completed"):
        response = client.post(
            "/train/patchtst/india?skip_snapshot=true",
            json={"universe": bucket.universe},
        )
        assert response.status_code == 202, response.text
        status = client.get("/train/status/" + response.json()["job_id"]).json()
        assert status["status"] == expected, status
        if expected == "failed":
            assert storage.read_metadata(version)["publication_pending"]
            assert storage.read_current_version() is None
    assert upload.call_count == 2
    forbidden_fit.assert_not_called()
    assert not storage.read_metadata(version)["publication_pending"]
    assert storage.read_current_version() == version

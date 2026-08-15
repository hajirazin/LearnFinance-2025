"""Tests for ``persist_forecaster_snapshot`` (canonical vs rejected)."""

from datetime import date
from unittest.mock import MagicMock

import torch
from sklearn.preprocessing import StandardScaler

from brain_api.routes.training.snapshot_persist import persist_forecaster_snapshot
from brain_api.storage.forecaster_snapshots import SnapshotLocalStorage


def _dummy_model_config():
    mock_model = MagicMock()
    mock_model.state_dict.return_value = {"weight": torch.tensor([1.0])}
    mock_config = MagicMock()
    mock_config.to_dict.return_value = {"hidden_size": 64}
    return mock_model, StandardScaler(), mock_config


def test_persist_healthy_writes_canonical_and_uploads(tmp_path) -> None:
    storage = SnapshotLocalStorage("lstm", base_path=tmp_path)
    storage.upload_snapshot_to_hf = MagicMock(return_value="user/repo")
    cutoff = date(2019, 12, 31)
    digest = "aaaaaaaaaaaa"
    model, scaler, config = _dummy_model_config()
    result = persist_forecaster_snapshot(
        snapshot_storage=storage,
        cutoff_date=cutoff,
        snapshot_digest=digest,
        model=model,
        feature_scaler=scaler,
        config=config,
        metadata={"metrics": {"train_loss": 0.01, "val_loss": 0.02}},
        train_loss=0.01,
        val_loss=0.02,
        snapshot_hf_repo="user/repo",
        log_prefix="[test]",
    )
    assert result.is_canonical is True
    assert result.failure_reasons == []
    assert storage.snapshot_exists(cutoff, digest)
    storage.upload_snapshot_to_hf.assert_called_once_with(cutoff, digest)


def test_persist_nan_val_loss_keeps_existing_canonical(tmp_path) -> None:
    storage = SnapshotLocalStorage("lstm", base_path=tmp_path)
    storage.upload_snapshot_to_hf = MagicMock()
    cutoff = date(2019, 12, 31)
    digest_a = "aaaaaaaaaaaa"
    digest_b = "bbbbbbbbbbbb"
    model, scaler, config = _dummy_model_config()
    storage.write_snapshot(
        cutoff_date=cutoff,
        snapshot_digest=digest_a,
        model=model,
        feature_scaler=scaler,
        config=config,
        metadata={"ok": True},
    )
    result = persist_forecaster_snapshot(
        snapshot_storage=storage,
        cutoff_date=cutoff,
        snapshot_digest=digest_b,
        model=model,
        feature_scaler=scaler,
        config=config,
        metadata={"metrics": {"train_loss": 0.01, "val_loss": float("nan")}},
        train_loss=0.01,
        val_loss=float("nan"),
        snapshot_hf_repo="user/repo",
        log_prefix="[test]",
    )
    assert result.is_canonical is False
    assert "val_loss is not finite" in result.failure_reasons
    dirs = storage.hashed_snapshot_dirs_for_cutoff(cutoff)
    assert len(dirs) == 1
    assert digest_a in dirs[0].name
    assert storage.rejected_snapshot_exists(cutoff, digest_b)
    assert storage.list_snapshots() == [cutoff]
    storage.upload_snapshot_to_hf.assert_not_called()
    rejected_meta = (result.path / "metadata.json").read_text()
    assert "failure_reasons" in rejected_meta


def test_persist_inf_val_loss_writes_rejected_only(tmp_path) -> None:
    storage = SnapshotLocalStorage("lstm", base_path=tmp_path)
    cutoff = date(2019, 12, 31)
    digest = "aaaaaaaaaaaa"
    model, scaler, config = _dummy_model_config()
    result = persist_forecaster_snapshot(
        snapshot_storage=storage,
        cutoff_date=cutoff,
        snapshot_digest=digest,
        model=model,
        feature_scaler=scaler,
        config=config,
        metadata={},
        train_loss=0.01,
        val_loss=float("inf"),
        snapshot_hf_repo=None,
        log_prefix="[test]",
    )
    assert result.is_canonical is False
    assert storage.list_snapshots() == []
    assert storage.rejected_snapshot_exists(cutoff, digest)


def test_persist_nonpositive_val_loss_writes_rejected(tmp_path) -> None:
    storage = SnapshotLocalStorage("lstm", base_path=tmp_path)
    cutoff = date(2019, 12, 31)
    digest = "aaaaaaaaaaaa"
    model, scaler, config = _dummy_model_config()
    result = persist_forecaster_snapshot(
        snapshot_storage=storage,
        cutoff_date=cutoff,
        snapshot_digest=digest,
        model=model,
        feature_scaler=scaler,
        config=config,
        metadata={},
        train_loss=0.01,
        val_loss=0.0,
        snapshot_hf_repo=None,
        log_prefix="[test]",
    )
    assert result.is_canonical is False
    assert any("val_loss must be > 0" in r for r in result.failure_reasons)
    assert storage.rejected_snapshot_exists(cutoff, digest)

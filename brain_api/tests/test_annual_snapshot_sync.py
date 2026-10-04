"""Annual snapshot parity: copy a one-sided December 31 digest, else train."""

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from brain_api.storage.forecaster_snapshots import SnapshotLocalStorage


class TestAnnualSnapshotSync:
    """Copy an existing December 31 digest. Train only when neither side has it."""

    def _run(self, storage, policy):
        from brain_api.routes.training.snapshot_phase import (
            _backfill_lstm_snapshots,
        )

        with (
            patch(
                "brain_api.routes.training.snapshot_phase.load_prices_yfinance",
                return_value={"AAPL": MagicMock()},
            ) as mock_load,
            patch(
                "brain_api.routes.training.snapshot_phase._filter_prices_by_cutoff",
                return_value={"AAPL": MagicMock()},
            ),
            patch(
                "brain_api.routes.training.snapshot_phase.build_dataset",
                return_value=MagicMock(X=[1]),
            ),
            patch(
                "brain_api.routes.training.snapshot_phase.train_model_pytorch",
            ) as mock_train,
            patch("brain_api.routes.training.snapshot_phase.gc"),
            patch("brain_api.routes.training.snapshot_phase.torch"),
        ):
            mock_train.return_value = MagicMock(
                train_loss=0.01,
                val_loss=0.02,
                best_epoch=1,
                stopped_epoch=1,
            )
            _backfill_lstm_snapshots(
                symbols=["AAPL"],
                config=MagicMock(to_dict=dict),
                start_date=date(2020, 1, 1),
                end_date=date(2020, 6, 1),
                snapshot_storage=storage,
                policy=policy,
            )
        return mock_load, mock_train

    @pytest.mark.parametrize("policy_name", ["local_first", "hf_first"])
    def test_local_only_uploads_and_does_not_train(self, policy_name):
        from brain_api.storage.policy import StoragePolicy

        policy = StoragePolicy(policy_name)
        storage = MagicMock(spec=SnapshotLocalStorage)
        storage.forecaster_type = "lstm_halal_new"
        storage.snapshot_exists.return_value = True
        storage._get_hf_repo.return_value = "user/repo"
        storage.snapshot_digest_exists_on_hf.return_value = False
        storage.upload_snapshot_to_hf.return_value = "user/repo"

        _mock_load, mock_train = self._run(storage, policy)

        storage.upload_snapshot_to_hf.assert_called_once()
        mock_train.assert_not_called()

    @pytest.mark.parametrize("policy_name", ["local_first", "hf_first"])
    def test_hugging_face_only_downloads_and_does_not_train(self, policy_name):
        from brain_api.storage.policy import StoragePolicy

        policy = StoragePolicy(policy_name)
        storage = MagicMock(spec=SnapshotLocalStorage)
        storage.forecaster_type = "lstm_halal_new"
        storage.snapshot_exists.return_value = False
        storage._get_hf_repo.return_value = "user/repo"
        storage.snapshot_digest_exists_on_hf.return_value = True
        storage.download_snapshot_from_hf.return_value = True

        _mock_load, mock_train = self._run(storage, policy)

        storage.download_snapshot_from_hf.assert_called_once()
        mock_train.assert_not_called()

    @pytest.mark.parametrize("policy_name", ["local_first", "hf_first"])
    def test_refused_download_aborts(self, policy_name):
        from brain_api.storage.policy import StoragePolicy

        policy = StoragePolicy(policy_name)
        storage = MagicMock(spec=SnapshotLocalStorage)
        storage.forecaster_type = "lstm_halal_new"
        storage.snapshot_exists.return_value = False
        storage._get_hf_repo.return_value = "user/repo"
        storage.snapshot_digest_exists_on_hf.return_value = True
        storage.download_snapshot_from_hf.return_value = False

        with pytest.raises(RuntimeError, match="refusing to retrain"):
            self._run(storage, policy)
        storage.upload_snapshot_to_hf.assert_not_called()

    @pytest.mark.parametrize("policy_name", ["local_first", "hf_first"])
    def test_neither_side_trains(self, policy_name):
        from brain_api.storage.policy import StoragePolicy

        policy = StoragePolicy(policy_name)
        storage = MagicMock(spec=SnapshotLocalStorage)
        storage.forecaster_type = "lstm_halal_new"
        storage.snapshot_exists.return_value = False
        storage._get_hf_repo.return_value = "user/repo"
        storage.snapshot_digest_exists_on_hf.return_value = False

        _mock_load, mock_train = self._run(storage, policy)

        mock_train.assert_called_once()
        storage.download_snapshot_from_hf.assert_not_called()
        storage.upload_snapshot_to_hf.assert_called_once()


@pytest.mark.parametrize(
    "bucket", ["lstm_halal_new", "patchtst_halal_new", "patchtst_nifty_shariah_500"]
)
@pytest.mark.parametrize("local_exists", [False, True])
def test_hf_listing_outage_aborts_before_copy_or_training(
    tmp_path, monkeypatch, bucket, local_exists
):
    """Exercise real storage: unknown HF inventory must not become 'missing'."""
    from brain_api.routes.training.snapshot_phase import _cutoffs_requiring_training

    storage = SnapshotLocalStorage(bucket, base_path=tmp_path)
    monkeypatch.setattr(storage, "_get_hf_repo", lambda: "user/repo")
    monkeypatch.setattr(storage, "snapshot_exists", lambda *args: local_exists)
    api = MagicMock()
    api.list_repo_refs.side_effect = ConnectionError("HF listing unavailable")
    monkeypatch.setattr(
        "brain_api.storage.forecaster_snapshots.snapshot_hf.HfApi", lambda **kwargs: api
    )
    upload = MagicMock()
    download = MagicMock()
    monkeypatch.setattr(storage, "upload_snapshot_to_hf", upload)
    monkeypatch.setattr(storage, "download_snapshot_from_hf", download)

    with pytest.raises(ConnectionError, match="HF listing unavailable"):
        _cutoffs_requiring_training(storage, {}, date(2020, 1, 1), date(2020, 6, 1))
    upload.assert_not_called()
    download.assert_not_called()


@pytest.mark.parametrize(
    "bucket", ["lstm_halal_new", "patchtst_halal_new", "patchtst_nifty_shariah_500"]
)
def test_confirmed_hf_snapshot_download_outage_aborts(tmp_path, monkeypatch, bucket):
    """The real download helper returns False on network errors; never retrain."""
    from brain_api.core.version import compute_snapshot_identity_hash
    from brain_api.routes.training.snapshot_phase import _cutoffs_requiring_training

    storage = SnapshotLocalStorage(bucket, base_path=tmp_path)
    monkeypatch.setattr(storage, "_get_hf_repo", lambda: "user/repo")
    cutoff = date(2019, 12, 31)
    digest = compute_snapshot_identity_hash(bucket, cutoff, {})
    api = MagicMock()
    branch = MagicMock()
    branch.name = f"snapshot-{cutoff}-{digest}"
    api.list_repo_refs.return_value.branches = [branch]
    monkeypatch.setattr(
        "brain_api.storage.forecaster_snapshots.snapshot_hf.HfApi", lambda **kwargs: api
    )
    download = MagicMock(side_effect=ConnectionError("HF download unavailable"))
    monkeypatch.setattr(
        "brain_api.storage.forecaster_snapshots.snapshot_hf.snapshot_download", download
    )

    with pytest.raises(RuntimeError, match="refusing to retrain"):
        _cutoffs_requiring_training(storage, {}, date(2020, 1, 1), date(2020, 6, 1))
    download.assert_called_once()
    assert not storage.snapshot_exists(cutoff, digest)


def test_rejected_annual_snapshot_is_retrained(tmp_path):
    """A rejected December 31 copy does not satisfy the backfill."""
    from sklearn.preprocessing import StandardScaler

    from brain_api.core.version import compute_snapshot_identity_hash
    from brain_api.routes.training.snapshot_phase import _backfill_lstm_snapshots

    storage = SnapshotLocalStorage("lstm_halal_new", base_path=tmp_path)
    config = MagicMock()
    config.to_dict.return_value = {"hidden": 16}
    cutoff = date(2019, 12, 31)
    digest = compute_snapshot_identity_hash("lstm_halal_new", cutoff, config.to_dict())
    model = MagicMock()
    model.state_dict.return_value = {}
    rejected = storage.write_rejected_snapshot(
        cutoff_date=cutoff,
        snapshot_digest=digest,
        model=model,
        feature_scaler=StandardScaler(),
        config=config,
        metadata={"failure_reasons": ["val_loss is not finite"]},
    )

    mock_result = MagicMock()
    mock_result.train_loss = 0.01
    mock_result.val_loss = 0.02
    mock_result.best_epoch = 1
    mock_result.stopped_epoch = 1
    mock_result.model.state_dict.return_value = {}
    mock_result.feature_scaler = StandardScaler()

    with (
        patch(
            "brain_api.routes.training.snapshot_phase.load_prices_yfinance",
            return_value={"AAPL": MagicMock()},
        ),
        patch(
            "brain_api.routes.training.snapshot_phase._filter_prices_by_cutoff",
            return_value={"AAPL": MagicMock()},
        ),
        patch(
            "brain_api.routes.training.snapshot_phase.build_dataset",
            return_value=MagicMock(X=[1]),
        ),
        patch(
            "brain_api.routes.training.snapshot_phase.train_model_pytorch",
            return_value=mock_result,
        ) as mock_train,
        patch("brain_api.routes.training.snapshot_phase.gc"),
        patch("brain_api.routes.training.snapshot_phase.torch"),
        patch.object(storage, "_get_hf_repo", return_value=None),
    ):
        _backfill_lstm_snapshots(
            symbols=["AAPL"],
            config=config,
            start_date=date(2020, 1, 1),
            end_date=date(2020, 6, 1),
            snapshot_storage=storage,
        )

    mock_train.assert_called_once()
    assert rejected.exists()

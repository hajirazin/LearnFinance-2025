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
    def test_refused_download_trains(self, policy_name):
        from brain_api.storage.policy import StoragePolicy

        policy = StoragePolicy(policy_name)
        storage = MagicMock(spec=SnapshotLocalStorage)
        storage.forecaster_type = "lstm_halal_new"
        storage.snapshot_exists.return_value = False
        storage._get_hf_repo.return_value = "user/repo"
        storage.snapshot_digest_exists_on_hf.return_value = True
        storage.download_snapshot_from_hf.return_value = False

        _mock_load, mock_train = self._run(storage, policy)

        mock_train.assert_called_once()

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

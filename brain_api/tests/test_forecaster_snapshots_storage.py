"""Tests for ``SnapshotLocalStorage`` and snapshot folder naming.

Covers:

* :class:`TestSnapshotLocalStorage` -- the storage class itself
  (init, write/list/exists, year-fallback lookup, metadata roundtrip).
* :class:`TestCreateSnapshotMetadata` -- the pure metadata factory.
* :class:`TestSnapshotFolderNaming` -- hashed-folder name parsing
  + the "second write removes sibling digests" invariant.
* :class:`TestPatchTSTSnapshots` -- bucket-name normalization + the
  unknown-bucket guardrail; lives here (not in the walk-forward or
  backfill files) because it exercises the storage class only.
"""

from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from brain_api.storage.forecaster_snapshots import (
    SnapshotLocalStorage,
    create_snapshot_metadata,
)

_TEST_SNAPSHOT_DIGEST = (
    "aaaaaaaaaaaa"  # 12 lowercase hex chars (matches compute_model_hash format)
)


class TestSnapshotLocalStorage:
    """Tests for SnapshotLocalStorage class."""

    def test_init_creates_storage(self, tmp_path):
        """Test storage initialization.

        Legacy short forecaster type ``"lstm"`` is normalised to the
        canonical bucket name ``"lstm_halal_new"`` so SAC's existing
        walk-forward callers keep working without a code change.
        """
        storage = SnapshotLocalStorage("lstm", base_path=tmp_path)
        assert storage.forecaster_type == "lstm_halal_new"
        assert storage.base_path == tmp_path

    def test_snapshot_path(self, tmp_path):
        """Hashed snapshots live under snapshot-{cutoff}-{digest}/."""

        storage = SnapshotLocalStorage("lstm", base_path=tmp_path)
        cutoff = date(2019, 12, 31)
        digest = _TEST_SNAPSHOT_DIGEST
        expected = (
            tmp_path / "models" / "lstm_halal_new" / f"snapshot-2019-12-31-{digest}"
        )
        assert storage._snapshot_path(cutoff, digest) == expected

    def test_snapshot_exists_false(self, tmp_path):
        """snapshot_exists checks the hashed folder basename."""

        storage = SnapshotLocalStorage("lstm", base_path=tmp_path)
        assert (
            storage.snapshot_exists(date(2019, 12, 31), _TEST_SNAPSHOT_DIGEST) is False
        )

    def test_list_snapshots_empty(self, tmp_path):
        """Test list_snapshots returns empty when no snapshots."""
        storage = SnapshotLocalStorage("lstm", base_path=tmp_path)
        assert storage.list_snapshots() == []

    def test_write_and_list_snapshot(self, tmp_path):
        """Test writing and listing a snapshot."""
        storage = SnapshotLocalStorage("lstm", base_path=tmp_path)
        cutoff = date(2019, 12, 31)

        mock_model = MagicMock()
        mock_model.state_dict.return_value = {"weight": torch.tensor([1.0])}
        mock_scaler = StandardScaler()
        mock_scaler.mean_ = np.array([0.0])
        mock_scaler.scale_ = np.array([1.0])
        mock_config = MagicMock()
        mock_config.to_dict.return_value = {"hidden_size": 64}
        metadata = {"test": "value"}

        snapshot_path = storage.write_snapshot(
            cutoff_date=cutoff,
            snapshot_digest=_TEST_SNAPSHOT_DIGEST,
            model=mock_model,
            feature_scaler=mock_scaler,
            config=mock_config,
            metadata=metadata,
        )

        assert snapshot_path == storage._snapshot_path(cutoff, _TEST_SNAPSHOT_DIGEST)
        assert (snapshot_path / "weights.pt").exists()
        assert (snapshot_path / "feature_scaler.pkl").exists()
        assert (snapshot_path / "config.json").exists()
        assert (snapshot_path / "metadata.json").exists()

        snapshots = storage.list_snapshots()
        assert len(snapshots) == 1
        assert snapshots[0] == cutoff

    def test_snapshot_exists_after_write(self, tmp_path):
        """Test snapshot_exists returns True after write."""
        storage = SnapshotLocalStorage("lstm", base_path=tmp_path)
        cutoff = date(2019, 12, 31)

        mock_model = MagicMock()
        mock_model.state_dict.return_value = {}
        mock_config = MagicMock()
        mock_config.to_dict.return_value = {}

        storage.write_snapshot(
            cutoff_date=cutoff,
            snapshot_digest=_TEST_SNAPSHOT_DIGEST,
            model=mock_model,
            feature_scaler=StandardScaler(),
            config=mock_config,
            metadata={},
        )

        assert storage.snapshot_exists(cutoff, _TEST_SNAPSHOT_DIGEST) is True

    def test_get_snapshot_for_year_exact_match(self, tmp_path):
        """Test get_snapshot_for_year with exact cutoff match."""
        storage = SnapshotLocalStorage("lstm", base_path=tmp_path)

        # Create snapshots for 2019 and 2020
        for year in [2019, 2020]:
            cutoff = date(year, 12, 31)
            mock_model = MagicMock()
            mock_model.state_dict.return_value = {}
            mock_config = MagicMock()
            mock_config.to_dict.return_value = {}
            storage.write_snapshot(
                cutoff_date=cutoff,
                snapshot_digest=_TEST_SNAPSHOT_DIGEST,
                model=mock_model,
                feature_scaler=StandardScaler(),
                config=mock_config,
                metadata={},
            )

        # Year 2020 should use 2019-12-31 snapshot
        assert storage.get_snapshot_for_year(2020) == date(2019, 12, 31)
        # Year 2021 should use 2020-12-31 snapshot
        assert storage.get_snapshot_for_year(2021) == date(2020, 12, 31)

    def test_get_snapshot_for_year_fallback(self, tmp_path):
        """Test get_snapshot_for_year falls back to closest snapshot."""
        storage = SnapshotLocalStorage("lstm", base_path=tmp_path)

        # Only create 2018 snapshot
        cutoff = date(2018, 12, 31)
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {}
        mock_config = MagicMock()
        mock_config.to_dict.return_value = {}
        storage.write_snapshot(
            cutoff_date=cutoff,
            snapshot_digest=_TEST_SNAPSHOT_DIGEST,
            model=mock_model,
            feature_scaler=StandardScaler(),
            config=mock_config,
            metadata={},
        )

        # Year 2020 should fall back to 2018-12-31 (no 2019-12-31 exists)
        assert storage.get_snapshot_for_year(2020) == date(2018, 12, 31)

    def test_get_snapshot_for_year_none(self, tmp_path):
        """Test get_snapshot_for_year returns None when no valid snapshot."""
        storage = SnapshotLocalStorage("lstm", base_path=tmp_path)

        # Only create 2020 snapshot
        cutoff = date(2020, 12, 31)
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {}
        mock_config = MagicMock()
        mock_config.to_dict.return_value = {}
        storage.write_snapshot(
            cutoff_date=cutoff,
            snapshot_digest=_TEST_SNAPSHOT_DIGEST,
            model=mock_model,
            feature_scaler=StandardScaler(),
            config=mock_config,
            metadata={},
        )

        # Year 2020 needs 2019-12-31 which doesn't exist
        assert storage.get_snapshot_for_year(2020) is None

    def test_read_metadata(self, tmp_path):
        """Test reading snapshot metadata."""
        storage = SnapshotLocalStorage("lstm", base_path=tmp_path)
        cutoff = date(2019, 12, 31)

        mock_model = MagicMock()
        mock_model.state_dict.return_value = {}
        mock_config = MagicMock()
        mock_config.to_dict.return_value = {}
        metadata = {"forecaster_type": "lstm", "test": "value"}

        storage.write_snapshot(
            cutoff_date=cutoff,
            snapshot_digest=_TEST_SNAPSHOT_DIGEST,
            model=mock_model,
            feature_scaler=StandardScaler(),
            config=mock_config,
            metadata=metadata,
        )

        read_meta = storage.read_metadata(cutoff)
        assert read_meta["forecaster_type"] == "lstm"
        assert read_meta["test"] == "value"

    def test_read_metadata_not_found(self, tmp_path):
        """Test reading metadata for non-existent snapshot."""
        storage = SnapshotLocalStorage("lstm", base_path=tmp_path)
        assert storage.read_metadata(date(2019, 12, 31)) is None


class TestCreateSnapshotMetadata:
    """Tests for create_snapshot_metadata function."""

    def test_creates_valid_metadata(self):
        """Test metadata creation."""
        mock_config = MagicMock()
        mock_config.to_dict.return_value = {"hidden_size": 64}

        metadata = create_snapshot_metadata(
            forecaster_type="lstm",
            cutoff_date=date(2019, 12, 31),
            data_window_start="2016-01-01",
            data_window_end="2019-12-31",
            symbols=["AAPL", "MSFT"],
            config=mock_config,
            train_loss=0.01,
            val_loss=0.02,
            best_epoch=3,
            stopped_epoch=18,
            config_symbols_hash="bbbbbbbbbbbb",
        )

        assert metadata["forecaster_type"] == "lstm"
        assert metadata["cutoff_date"] == "2019-12-31"
        assert metadata["data_window"]["start"] == "2016-01-01"
        assert metadata["data_window"]["end"] == "2019-12-31"
        assert metadata["symbols"] == ["AAPL", "MSFT"]
        assert metadata["config_symbols_hash"] == "bbbbbbbbbbbb"
        assert metadata["metrics"]["train_loss"] == 0.01
        assert metadata["metrics"]["val_loss"] == 0.02
        assert metadata["metrics"]["best_epoch"] == 3
        assert metadata["metrics"]["stopped_epoch"] == 18
        assert "training_timestamp" in metadata


class TestSnapshotFolderNaming:
    """Naming + parsing helpers for hashed snapshot dirs / HF branches."""

    def test_parse_hashed_folder_name_accept(self) -> None:
        from brain_api.storage.forecaster_snapshots.snapshot_layout import (
            parse_hashed_snapshot_folder_name,
        )

        name = "snapshot-2019-12-31-abcdef012345"
        cutoff, digest = parse_hashed_snapshot_folder_name(name)
        assert cutoff == date(2019, 12, 31)
        assert digest == "abcdef012345"

    def test_parse_legacy_flat_folder_name_rejected(self) -> None:
        from brain_api.storage.forecaster_snapshots.snapshot_layout import (
            parse_hashed_snapshot_folder_name,
        )

        assert parse_hashed_snapshot_folder_name("snapshot-2019-12-31") is None

    def test_write_removes_other_digest_for_same_cutoff(self, tmp_path) -> None:
        """Second write deletes sibling ``snapshot-{{cut}}-*`` dirs."""

        storage = SnapshotLocalStorage("lstm", base_path=tmp_path)
        cutoff = date(2019, 12, 31)
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {}
        mock_cfg = MagicMock()
        mock_cfg.to_dict.return_value = {}
        scaler = StandardScaler()

        digest_a = "aaaaaaaaaaaa"
        digest_b = "bbbbbbbbbbbb"
        storage.write_snapshot(
            cutoff_date=cutoff,
            snapshot_digest=digest_a,
            model=mock_model,
            feature_scaler=scaler,
            config=mock_cfg,
            metadata={},
        )
        legacy_flat = storage._models_path / f"snapshot-{cutoff.isoformat()}"
        legacy_flat.mkdir()
        assert len(storage.hashed_snapshot_dirs_for_cutoff(cutoff)) == 1

        storage.write_snapshot(
            cutoff_date=cutoff,
            snapshot_digest=digest_b,
            model=mock_model,
            feature_scaler=scaler,
            config=mock_cfg,
            metadata={},
        )
        dirs = storage.hashed_snapshot_dirs_for_cutoff(cutoff)
        assert len(dirs) == 1
        assert digest_b in dirs[0].name
        assert legacy_flat.exists()


class TestPatchTSTSnapshots:
    """Tests specific to PatchTST snapshot handling."""

    def test_patchtst_storage_type(self, tmp_path):
        """Legacy ``"patchtst"`` resolves to the canonical bucket name."""
        storage = SnapshotLocalStorage("patchtst", base_path=tmp_path)
        assert storage.forecaster_type == "patchtst_halal_new"
        expected = tmp_path / "models" / "patchtst_halal_new"
        assert storage._models_path == expected

    def test_patchtst_india_storage_type(self, tmp_path):
        """India PatchTST snapshots live under their own bucket directory."""
        storage = SnapshotLocalStorage("patchtst_nifty_shariah_500", base_path=tmp_path)
        assert storage.forecaster_type == "patchtst_nifty_shariah_500"
        expected = tmp_path / "models" / "patchtst_nifty_shariah_500"
        assert storage._models_path == expected

    def test_unknown_forecaster_type_raises(self, tmp_path):
        """Unknown bucket names fail fast (no silent fallback)."""
        with pytest.raises(ValueError):
            SnapshotLocalStorage("not_a_bucket", base_path=tmp_path)

    def test_patchtst_snapshot_path(self, tmp_path):
        """PatchTST snapshot path uses hashed layout under the canonical bucket."""

        storage = SnapshotLocalStorage("patchtst", base_path=tmp_path)
        cutoff = date(2019, 12, 31)
        digest = _TEST_SNAPSHOT_DIGEST
        expected = (
            tmp_path / "models" / "patchtst_halal_new" / f"snapshot-2019-12-31-{digest}"
        )
        assert storage._snapshot_path(cutoff, digest) == expected

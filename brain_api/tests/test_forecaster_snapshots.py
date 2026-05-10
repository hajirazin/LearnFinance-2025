"""Tests for forecaster snapshot training, storage, and walk-forward inference."""

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from brain_api.core.portfolio_rl.walkforward import (
    build_forecast_features,
)
from brain_api.storage.forecaster_snapshots import (
    LSTMSnapshotArtifacts,
    PatchTSTSnapshotArtifacts,
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


class TestWalkForwardForecasts:
    """Tests for walk-forward forecast generation."""

    def test_build_forecast_features_raises_on_missing_snapshots(self, tmp_path):
        """Test build_forecast_features raises when snapshots are missing."""
        from brain_api.core.portfolio_rl.walkforward import SnapshotUnavailableError

        n_weeks = 52
        weekly_prices = {
            "AAPL": np.linspace(100, 150, n_weeks),
        }
        weekly_dates = pd.date_range("2020-01-06", periods=n_weeks, freq="W-MON")

        with (
            patch(
                "brain_api.storage.forecaster_snapshots.local.DEFAULT_DATA_PATH",
                tmp_path,
            ),
            # Block the HuggingFace fallback -- the developer's local
            # ``.env`` may set ``HF_LSTM_HALAL_NEW_MODEL_REPO`` (it does in
            # this repo), and a populated remote repo would otherwise
            # satisfy the snapshot lookup and mask the missing-snapshot
            # error this test is meant to assert.
            patch.object(
                SnapshotLocalStorage,
                "_get_hf_repo",
                return_value=None,
            ),
            pytest.raises(SnapshotUnavailableError),
        ):
            build_forecast_features(
                weekly_prices,
                weekly_dates,
                ["AAPL"],
                forecaster_type="lstm",
            )

    def test_build_forecast_features_delegates_to_generate(self):
        """Test build_forecast_features calls generate_walkforward_forecasts."""
        n_weeks = 52
        weekly_prices = {
            "AAPL": np.linspace(100, 150, n_weeks),
        }
        weekly_dates = pd.date_range("2020-01-06", periods=n_weeks, freq="W-MON")

        with patch(
            "brain_api.core.portfolio_rl.walkforward.generate_walkforward_forecasts"
        ) as mock:
            mock.return_value = {"AAPL": np.zeros(n_weeks - 1)}
            forecasts = build_forecast_features(
                weekly_prices,
                weekly_dates,
                ["AAPL"],
                forecaster_type="lstm",
            )

        mock.assert_called_once()
        assert "AAPL" in forecasts

    def test_generate_forecasts_empty_prices(self):
        """Test forecast generation with empty data returns empty."""
        from brain_api.core.portfolio_rl.walkforward import (
            generate_walkforward_forecasts,
        )

        forecasts = generate_walkforward_forecasts({}, pd.DatetimeIndex([]), [], "lstm")
        assert forecasts == {}


class TestSnapshotIntegration:
    """Tests for integrated snapshot functionality in training endpoints."""

    def test_snapshots_available_helper(self, tmp_path):
        """Test _snapshots_available helper function."""
        from brain_api.routes.training import _snapshots_available

        # No snapshots exist initially
        with patch(
            "brain_api.routes.training.dependencies.SnapshotLocalStorage"
        ) as mock_storage:
            mock_instance = MagicMock()
            mock_instance.list_snapshots.return_value = []
            mock_storage.return_value = mock_instance
            assert _snapshots_available("lstm") is False

        # After adding snapshots
        with patch(
            "brain_api.routes.training.dependencies.SnapshotLocalStorage"
        ) as mock_storage:
            mock_instance = MagicMock()
            mock_instance.list_snapshots.return_value = [date(2019, 12, 31)]
            mock_storage.return_value = mock_instance
            assert _snapshots_available("lstm") is True

    def test_backfill_functions_exist(self):
        """Test that backfill helper functions exist."""
        from brain_api.routes.training import (
            _backfill_lstm_snapshots,
            _backfill_patchtst_snapshots,
        )

        # Just verify they can be imported
        assert callable(_backfill_lstm_snapshots)
        assert callable(_backfill_patchtst_snapshots)


class TestBackfillSnapshotRange:
    """Tests that backfill functions create snapshots for the full RL window."""

    def test_lstm_backfill_creates_snapshots_for_full_rl_range(self):
        """Verify _backfill_lstm_snapshots covers start_year-1 .. end_year-1.

        With start_date=2016-01-01, end_date=2025-12-26 (a Friday), the
        backfill must create snapshots 2015-12-31 through 2024-12-31
        (10 snapshots), and download prices starting from 2011-01-01
        (bootstrap_years=4 before 2015).
        """
        from brain_api.routes.training.snapshot_phase import (
            _backfill_lstm_snapshots,
        )

        mock_storage = MagicMock(spec=SnapshotLocalStorage)
        mock_storage.snapshot_exists_anywhere.return_value = False
        mock_storage.forecaster_type = "lstm_halal_new"

        mock_prices = {"AAPL": MagicMock(), "MSFT": MagicMock()}
        mock_dataset = MagicMock()
        mock_dataset.X = [1]  # non-empty
        mock_result = MagicMock()
        mock_result.train_loss = 0.01
        mock_result.val_loss = 0.02

        with (
            patch(
                "brain_api.routes.training.snapshot_phase.load_prices_yfinance",
                return_value=mock_prices,
            ) as mock_load,
            patch(
                "brain_api.routes.training.snapshot_phase.build_dataset",
                return_value=mock_dataset,
            ),
            patch(
                "brain_api.routes.training.snapshot_phase.train_model_pytorch",
                return_value=mock_result,
            ),
            patch(
                "brain_api.routes.training.snapshot_phase._filter_prices_by_cutoff",
                return_value=mock_prices,
            ),
            patch("brain_api.routes.training.snapshot_phase.gc"),
            patch("brain_api.routes.training.snapshot_phase.torch"),
        ):
            _backfill_lstm_snapshots(
                symbols=["AAPL", "MSFT"],
                config=MagicMock(to_dict=dict),
                start_date=date(2016, 1, 1),
                end_date=date(2025, 12, 26),
                snapshot_storage=mock_storage,
            )

        # Price data should start from 2011-01-01 (2016-1-4 = 2011)
        load_call_args = mock_load.call_args
        assert load_call_args[0][1] == date(2011, 1, 1)

        # Should write snapshots for 2015-12-31 through 2024-12-31
        write_calls = mock_storage.write_snapshot.call_args_list
        written_cutoffs = [c.kwargs["cutoff_date"] for c in write_calls]
        expected = [date(y, 12, 31) for y in range(2015, 2025)]
        assert written_cutoffs == expected

    def test_patchtst_backfill_creates_snapshots_for_full_rl_range(self):
        """Verify _backfill_patchtst_snapshots covers start_year-1 .. end_year-1."""
        from brain_api.routes.training.snapshot_phase import (
            _backfill_patchtst_snapshots,
        )

        mock_storage = MagicMock(spec=SnapshotLocalStorage)
        mock_storage.snapshot_exists_anywhere.return_value = False
        mock_storage.forecaster_type = "patchtst_halal_new"

        mock_prices = {"AAPL": MagicMock(), "MSFT": MagicMock()}
        mock_dataset = MagicMock()
        mock_dataset.X = [1]
        mock_result = MagicMock()
        mock_result.train_loss = 0.01
        mock_result.val_loss = 0.02

        with (
            patch(
                "brain_api.routes.training.snapshot_phase.patchtst_load_prices",
                return_value=mock_prices,
            ) as mock_load,
            patch(
                "brain_api.routes.training.snapshot_phase._filter_prices_by_cutoff",
                return_value=mock_prices,
            ),
            patch(
                "brain_api.routes.training.snapshot_phase.align_multivariate_data",
                return_value={"AAPL": MagicMock()},
            ),
            patch(
                "brain_api.routes.training.snapshot_phase.patchtst_build_dataset",
                return_value=mock_dataset,
            ),
            patch(
                "brain_api.routes.training.snapshot_phase.patchtst_train_model",
                return_value=mock_result,
            ),
        ):
            _backfill_patchtst_snapshots(
                symbols=["AAPL", "MSFT"],
                config=MagicMock(to_dict=dict),
                start_date=date(2016, 1, 1),
                end_date=date(2025, 12, 26),
                snapshot_storage=mock_storage,
            )

        # Price data should start from 2011-01-01
        load_call_args = mock_load.call_args
        assert load_call_args[0][1] == date(2011, 1, 1)

        # Should write snapshots for 2015-12-31 through 2024-12-31
        write_calls = mock_storage.write_snapshot.call_args_list
        written_cutoffs = [c.kwargs["cutoff_date"] for c in write_calls]
        expected = [date(y, 12, 31) for y in range(2015, 2025)]
        assert written_cutoffs == expected

    def test_lstm_backfill_skips_existing_snapshots(self):
        """Verify backfill skips snapshots that already exist."""
        from brain_api.routes.training.snapshot_phase import (
            _backfill_lstm_snapshots,
        )

        mock_storage = MagicMock(spec=SnapshotLocalStorage)
        mock_storage.forecaster_type = "lstm_halal_new"

        # Simulate: 2015-12-31 exists, all others don't
        def exists_side_effect(
            cutoff_date, _snapshot_digest, *, check_hf=False
        ) -> bool:
            return cutoff_date == date(2015, 12, 31)

        mock_storage.snapshot_exists_anywhere.side_effect = exists_side_effect

        mock_prices = {"AAPL": MagicMock()}
        mock_dataset = MagicMock()
        mock_dataset.X = [1]
        mock_result = MagicMock()
        mock_result.train_loss = 0.01
        mock_result.val_loss = 0.02

        with (
            patch(
                "brain_api.routes.training.snapshot_phase.load_prices_yfinance",
                return_value=mock_prices,
            ),
            patch(
                "brain_api.routes.training.snapshot_phase.build_dataset",
                return_value=mock_dataset,
            ),
            patch(
                "brain_api.routes.training.snapshot_phase.train_model_pytorch",
                return_value=mock_result,
            ),
            patch(
                "brain_api.routes.training.snapshot_phase._filter_prices_by_cutoff",
                return_value=mock_prices,
            ),
            patch("brain_api.routes.training.snapshot_phase.gc"),
            patch("brain_api.routes.training.snapshot_phase.torch"),
        ):
            _backfill_lstm_snapshots(
                symbols=["AAPL"],
                config=MagicMock(to_dict=dict),
                start_date=date(2016, 1, 1),
                end_date=date(2025, 12, 26),
                snapshot_storage=mock_storage,
            )

        # 2015-12-31 should be skipped, so 9 snapshots written (not 10)
        write_calls = mock_storage.write_snapshot.call_args_list
        written_cutoffs = [c.kwargs["cutoff_date"] for c in write_calls]
        assert date(2015, 12, 31) not in written_cutoffs
        assert len(written_cutoffs) == 9

    def test_lstm_backfill_metadata_uses_extended_start(self):
        """Verify snapshot metadata records the extended data_window_start."""
        from brain_api.routes.training.snapshot_phase import (
            _backfill_lstm_snapshots,
        )

        mock_storage = MagicMock(spec=SnapshotLocalStorage)
        mock_storage.snapshot_exists_anywhere.return_value = False
        mock_storage.forecaster_type = "lstm_halal_new"

        mock_prices = {"AAPL": MagicMock()}
        mock_dataset = MagicMock()
        mock_dataset.X = [1]
        mock_result = MagicMock()
        mock_result.train_loss = 0.01
        mock_result.val_loss = 0.02

        captured_metadata = []

        with (
            patch(
                "brain_api.routes.training.snapshot_phase.load_prices_yfinance",
                return_value=mock_prices,
            ),
            patch(
                "brain_api.routes.training.snapshot_phase.build_dataset",
                return_value=mock_dataset,
            ),
            patch(
                "brain_api.routes.training.snapshot_phase.train_model_pytorch",
                return_value=mock_result,
            ),
            patch(
                "brain_api.routes.training.snapshot_phase._filter_prices_by_cutoff",
                return_value=mock_prices,
            ),
            patch(
                "brain_api.routes.training.snapshot_phase.create_snapshot_metadata",
                side_effect=lambda **kw: (captured_metadata.append(kw) or {}),
            ),
            patch("brain_api.routes.training.snapshot_phase.gc"),
            patch("brain_api.routes.training.snapshot_phase.torch"),
        ):
            _backfill_lstm_snapshots(
                symbols=["AAPL"],
                config=MagicMock(to_dict=dict),
                start_date=date(2016, 1, 1),
                end_date=date(2025, 12, 26),
                snapshot_storage=mock_storage,
            )

        # All metadata entries should have data_window_start = 2011-01-01
        for meta_kwargs in captured_metadata:
            assert meta_kwargs["data_window_start"] == "2011-01-01"


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


class TestSnapshotInferenceHelpers:
    """Tests for snapshot inference helper functions."""

    def test_lstm_inference_raises_without_weekly_dates(self):
        """Test LSTM inference raises when weekly_dates is missing."""
        from brain_api.core.portfolio_rl.walkforward import (
            SnapshotInferenceError,
            _run_lstm_snapshot_inference,
        )

        mock_config = MagicMock()
        mock_config.sequence_length = 20
        mock_config.use_returns = True

        artifacts = LSTMSnapshotArtifacts(
            config=mock_config,
            feature_scaler=StandardScaler(),
            model=MagicMock(),
            cutoff_date=date(2019, 12, 31),
        )

        year_indices = [5, 6, 7]

        with pytest.raises(SnapshotInferenceError):
            _run_lstm_snapshot_inference(
                artifacts,
                year_indices,
                weekly_dates=None,
                symbol="TEST",
            )

    def test_patchtst_inference_raises_without_weekly_dates(self):
        """Test PatchTST inference raises when weekly_dates is missing."""
        from brain_api.core.portfolio_rl.walkforward import (
            SnapshotInferenceError,
            _run_patchtst_snapshot_inference,
        )

        mock_config = MagicMock()
        mock_config.context_length = 20

        artifacts = PatchTSTSnapshotArtifacts(
            config=mock_config,
            feature_scaler=StandardScaler(),
            model=MagicMock(),
            cutoff_date=date(2019, 12, 31),
        )

        year_indices = [5, 6, 7]

        with pytest.raises(SnapshotInferenceError):
            _run_patchtst_snapshot_inference(
                artifacts,
                year_indices,
                weekly_dates=None,
                symbol="TEST",
            )


class TestResolveCheckHF:
    """Truth-table for the policy translator that drives ``check_hf``.

    The four rows below are the entire decision matrix used by every
    snapshot existence-check call site (``_run_*_snapshot_phase``,
    ``_backfill_*_snapshots``, ``count_missing_snapshots``).
    Regressions here corrupt every downstream policy decision.
    """

    def test_local_first_no_hf_repo_returns_false(self):
        """``local_first`` + no HF repo: skip HF entirely (local-only)."""
        from brain_api.core.forecaster_snapshot_identity import _resolve_check_hf
        from brain_api.storage.policy import StoragePolicy

        snapshot_storage = MagicMock()
        snapshot_storage._get_hf_repo.return_value = None

        assert (
            _resolve_check_hf(
                snapshot_storage=snapshot_storage,
                policy=StoragePolicy.LOCAL_FIRST,
            )
            is False
        )

    def test_local_first_with_hf_repo_returns_true(self):
        """``local_first`` + HF repo: HF is the fallback for wiped local cache."""
        from brain_api.core.forecaster_snapshot_identity import _resolve_check_hf
        from brain_api.storage.policy import StoragePolicy

        snapshot_storage = MagicMock()
        snapshot_storage._get_hf_repo.return_value = "user/repo"

        assert (
            _resolve_check_hf(
                snapshot_storage=snapshot_storage,
                policy=StoragePolicy.LOCAL_FIRST,
            )
            is True
        )

    def test_hf_first_with_hf_repo_returns_true(self):
        """``hf_first`` + HF repo: consult HF first."""
        from brain_api.core.forecaster_snapshot_identity import _resolve_check_hf
        from brain_api.storage.policy import StoragePolicy

        snapshot_storage = MagicMock()
        snapshot_storage._get_hf_repo.return_value = "user/repo"

        assert (
            _resolve_check_hf(
                snapshot_storage=snapshot_storage,
                policy=StoragePolicy.HF_FIRST,
            )
            is True
        )

    def test_hf_first_no_hf_repo_raises_storage_policy_error(self):
        """``hf_first`` + no HF repo: must fail loudly (AGENTS.md rule #1).

        Per the no-silent-fallback rule: the operator chose ``hf_first``
        and there's no HF endpoint to consult; degrading to local-only
        would silently violate the chosen policy.
        """
        from brain_api.core.forecaster_snapshot_identity import _resolve_check_hf
        from brain_api.storage.policy import StoragePolicy, StoragePolicyError

        snapshot_storage = MagicMock()
        snapshot_storage._get_hf_repo.return_value = None
        snapshot_storage.forecaster_type = "lstm_halal_new"

        with pytest.raises(StoragePolicyError) as excinfo:
            _resolve_check_hf(
                snapshot_storage=snapshot_storage,
                policy=StoragePolicy.HF_FIRST,
            )
        msg = str(excinfo.value)
        assert "hf_first" in msg
        assert "lstm_halal_new" in msg


class TestCountMissingSnapshots:
    """Tests for the read-side mirror of the backfill loops.

    These tests pin the digest formulas (math correctness, AGENTS.md
    rule #2) by independently computing what the backfill loops would
    use and asserting the helper's existence-check calls match
    bit-for-bit.
    """

    @staticmethod
    def _expected_digests(
        forecaster_type: str,
        train_window: tuple[date, date],
        symbols: list[str],
        config_dict: dict,
    ) -> tuple[str, list[tuple[date, str]]]:
        """Return ``(end_window_digest, [(historical_cutoff, digest), ...])``.

        Mirrors ``count_missing_snapshots`` and the legacy backfill
        loops bit-for-bit. Any drift here means the helper has
        diverged from the trainer code.
        """
        from brain_api.core.version import compute_model_hash

        start_date, end_date = train_window
        end_window = compute_model_hash(
            forecaster_type, start_date, end_date, symbols, config_dict
        )

        bootstrap_years = 4
        first_snapshot_year = start_date.year - 1
        snapshot_data_start = date(first_snapshot_year - bootstrap_years, 1, 1)
        historical = []
        for year in range(first_snapshot_year, end_date.year):
            cutoff = date(year, 12, 31)
            digest = compute_model_hash(
                forecaster_type,
                snapshot_data_start,
                cutoff,
                symbols,
                config_dict,
            )
            historical.append((cutoff, digest))
        return end_window, historical

    def _build_storage(self, *, hf_repo: str | None, exists_map: dict) -> MagicMock:
        """Mock storage that returns ``exists_map.get((cutoff, digest), False)``."""
        storage = MagicMock(spec=SnapshotLocalStorage)
        storage.forecaster_type = "lstm_halal_new"
        storage._get_hf_repo.return_value = hf_repo

        def exists_side_effect(cutoff, digest, *, check_hf=False):
            return exists_map.get((cutoff, digest), False)

        storage.snapshot_exists_anywhere.side_effect = exists_side_effect
        return storage

    def test_all_present_returns_empty_inventory(self):
        from brain_api.core.forecaster_snapshot_identity import (
            count_missing_snapshots,
        )
        from brain_api.storage.policy import StoragePolicy

        train_window = (date(2016, 1, 1), date(2025, 12, 26))
        symbols = ["AAPL"]
        config_dict = {"k": "v"}
        end_window, historical = self._expected_digests(
            "lstm_halal_new", train_window, symbols, config_dict
        )
        exists_map = {(train_window[1], end_window): True}
        for cutoff, digest in historical:
            exists_map[(cutoff, digest)] = True
        storage = self._build_storage(hf_repo=None, exists_map=exists_map)

        inventory = count_missing_snapshots(
            forecaster_type="lstm_halal_new",
            train_window=train_window,
            symbols=symbols,
            config_dict=config_dict,
            snapshot_storage=storage,
            policy=StoragePolicy.LOCAL_FIRST,
        )
        assert inventory.is_empty
        assert inventory.total_missing == 0
        assert inventory.end_window_cutoff is None
        assert inventory.historical_cutoffs == ()

    def test_end_window_only_missing(self):
        from brain_api.core.forecaster_snapshot_identity import (
            count_missing_snapshots,
        )
        from brain_api.storage.policy import StoragePolicy

        train_window = (date(2016, 1, 1), date(2025, 12, 26))
        symbols = ["AAPL"]
        config_dict = {"k": "v"}
        _end_window, historical = self._expected_digests(
            "lstm_halal_new", train_window, symbols, config_dict
        )
        exists_map = dict.fromkeys(historical, True)
        storage = self._build_storage(hf_repo=None, exists_map=exists_map)

        inventory = count_missing_snapshots(
            forecaster_type="lstm_halal_new",
            train_window=train_window,
            symbols=symbols,
            config_dict=config_dict,
            snapshot_storage=storage,
            policy=StoragePolicy.LOCAL_FIRST,
        )
        assert not inventory.is_empty
        assert inventory.total_missing == 1
        assert inventory.end_window_cutoff == train_window[1]
        assert inventory.historical_cutoffs == ()

    def test_historical_only_missing(self):
        from brain_api.core.forecaster_snapshot_identity import (
            count_missing_snapshots,
        )
        from brain_api.storage.policy import StoragePolicy

        train_window = (date(2016, 1, 1), date(2025, 12, 26))
        symbols = ["AAPL"]
        config_dict = {"k": "v"}
        end_window, historical = self._expected_digests(
            "lstm_halal_new", train_window, symbols, config_dict
        )
        exists_map = {(train_window[1], end_window): True}
        # Mark all but the first historical as present
        for cutoff, digest in historical[1:]:
            exists_map[(cutoff, digest)] = True
        storage = self._build_storage(hf_repo=None, exists_map=exists_map)

        inventory = count_missing_snapshots(
            forecaster_type="lstm_halal_new",
            train_window=train_window,
            symbols=symbols,
            config_dict=config_dict,
            snapshot_storage=storage,
            policy=StoragePolicy.LOCAL_FIRST,
        )
        assert inventory.end_window_cutoff is None
        assert inventory.historical_cutoffs == (historical[0][0],)
        assert inventory.total_missing == 1

    def test_mixed_missing(self):
        from brain_api.core.forecaster_snapshot_identity import (
            count_missing_snapshots,
        )
        from brain_api.storage.policy import StoragePolicy

        train_window = (date(2016, 1, 1), date(2025, 12, 26))
        symbols = ["AAPL"]
        config_dict = {"k": "v"}
        _end_window, historical = self._expected_digests(
            "lstm_halal_new", train_window, symbols, config_dict
        )
        # End-window missing, plus the first 2 historical missing
        exists_map = dict.fromkeys(historical[2:], True)
        storage = self._build_storage(hf_repo=None, exists_map=exists_map)

        inventory = count_missing_snapshots(
            forecaster_type="lstm_halal_new",
            train_window=train_window,
            symbols=symbols,
            config_dict=config_dict,
            snapshot_storage=storage,
            policy=StoragePolicy.LOCAL_FIRST,
        )
        assert inventory.end_window_cutoff == train_window[1]
        assert inventory.historical_cutoffs == (
            historical[0][0],
            historical[1][0],
        )
        assert inventory.total_missing == 3

    def test_all_missing(self):
        from brain_api.core.forecaster_snapshot_identity import (
            count_missing_snapshots,
        )
        from brain_api.storage.policy import StoragePolicy

        train_window = (date(2016, 1, 1), date(2025, 12, 26))
        symbols = ["AAPL"]
        config_dict = {"k": "v"}
        _end_window, historical = self._expected_digests(
            "lstm_halal_new", train_window, symbols, config_dict
        )
        storage = self._build_storage(hf_repo=None, exists_map={})

        inventory = count_missing_snapshots(
            forecaster_type="lstm_halal_new",
            train_window=train_window,
            symbols=symbols,
            config_dict=config_dict,
            snapshot_storage=storage,
            policy=StoragePolicy.LOCAL_FIRST,
        )
        assert inventory.end_window_cutoff == train_window[1]
        assert inventory.historical_cutoffs == tuple(c for c, _ in historical)
        assert inventory.total_missing == 1 + len(historical)

    def test_hf_first_with_repo_propagates_check_hf_true(self):
        from brain_api.core.forecaster_snapshot_identity import (
            count_missing_snapshots,
        )
        from brain_api.storage.policy import StoragePolicy

        train_window = (date(2016, 1, 1), date(2025, 12, 26))
        symbols = ["AAPL"]
        config_dict = {"k": "v"}
        storage = self._build_storage(hf_repo="user/repo", exists_map={})

        count_missing_snapshots(
            forecaster_type="lstm_halal_new",
            train_window=train_window,
            symbols=symbols,
            config_dict=config_dict,
            snapshot_storage=storage,
            policy=StoragePolicy.HF_FIRST,
        )

        # Every existence call should have been made with check_hf=True
        for call in storage.snapshot_exists_anywhere.call_args_list:
            assert call.kwargs["check_hf"] is True

    def test_local_first_no_repo_skips_hf(self):
        from brain_api.core.forecaster_snapshot_identity import (
            count_missing_snapshots,
        )
        from brain_api.storage.policy import StoragePolicy

        train_window = (date(2016, 1, 1), date(2025, 12, 26))
        symbols = ["AAPL"]
        config_dict = {"k": "v"}
        storage = self._build_storage(hf_repo=None, exists_map={})

        count_missing_snapshots(
            forecaster_type="lstm_halal_new",
            train_window=train_window,
            symbols=symbols,
            config_dict=config_dict,
            snapshot_storage=storage,
            policy=StoragePolicy.LOCAL_FIRST,
        )

        for call in storage.snapshot_exists_anywhere.call_args_list:
            assert call.kwargs["check_hf"] is False

    def test_hf_first_no_repo_raises(self):
        from brain_api.core.forecaster_snapshot_identity import (
            count_missing_snapshots,
        )
        from brain_api.storage.policy import StoragePolicy, StoragePolicyError

        train_window = (date(2016, 1, 1), date(2025, 12, 26))
        symbols = ["AAPL"]
        config_dict = {"k": "v"}
        storage = self._build_storage(hf_repo=None, exists_map={})

        with pytest.raises(StoragePolicyError):
            count_missing_snapshots(
                forecaster_type="lstm_halal_new",
                train_window=train_window,
                symbols=symbols,
                config_dict=config_dict,
                snapshot_storage=storage,
                policy=StoragePolicy.HF_FIRST,
            )

    def test_default_policy_resolves_from_env(self, monkeypatch):
        """``policy=None`` resolves to ``get_storage_policy()`` (i.e. env)."""
        from brain_api.core.forecaster_snapshot_identity import (
            count_missing_snapshots,
        )

        monkeypatch.setenv("STORAGE_BACKEND", "local_first")

        train_window = (date(2016, 1, 1), date(2025, 12, 26))
        symbols = ["AAPL"]
        config_dict = {"k": "v"}
        storage = self._build_storage(hf_repo=None, exists_map={})

        count_missing_snapshots(
            forecaster_type="lstm_halal_new",
            train_window=train_window,
            symbols=symbols,
            config_dict=config_dict,
            snapshot_storage=storage,
        )

        for call in storage.snapshot_exists_anywhere.call_args_list:
            assert call.kwargs["check_hf"] is False

    def test_digest_inputs_match_backfill_formula(self):
        """Math-correctness regression: the helper MUST call
        ``compute_model_hash`` with the exact same inputs as the
        backfill loops. If this drifts, every downstream snapshot
        decision silently breaks (AGENTS.md rule #2)."""
        from brain_api.core.forecaster_snapshot_identity import (
            count_missing_snapshots,
        )
        from brain_api.storage.policy import StoragePolicy

        train_window = (date(2016, 1, 1), date(2025, 12, 26))
        symbols = ["AAPL", "MSFT"]
        config_dict = {"hidden": 16}
        storage = self._build_storage(hf_repo=None, exists_map={})

        count_missing_snapshots(
            forecaster_type="lstm_halal_new",
            train_window=train_window,
            symbols=symbols,
            config_dict=config_dict,
            snapshot_storage=storage,
            policy=StoragePolicy.LOCAL_FIRST,
        )

        end_window_digest, historical = self._expected_digests(
            "lstm_halal_new", train_window, symbols, config_dict
        )
        observed = [
            (call.args[0], call.args[1])
            for call in storage.snapshot_exists_anywhere.call_args_list
        ]
        assert observed[0] == (train_window[1], end_window_digest)
        assert observed[1:] == [(c, d) for c, d in historical]


class TestBackfillLoopsRespectPolicy:
    """Regression coverage for the ``policy``-aware refactor of the
    backfill loops (closes the latent local-only bug found while
    auditing the original branchless ``check_hf = hf_repo is not None``).
    """

    def _common_patches(self, mock_storage):
        """Common patches for backfill snapshot tests (LSTM + PatchTST)."""
        from contextlib import ExitStack

        stack = ExitStack()
        stack.enter_context(
            patch(
                "brain_api.routes.training.snapshot_phase.load_prices_yfinance",
                return_value={"AAPL": MagicMock()},
            )
        )
        stack.enter_context(
            patch(
                "brain_api.routes.training.snapshot_phase._filter_prices_by_cutoff",
                return_value={"AAPL": MagicMock()},
            )
        )
        mock_dataset = MagicMock()
        mock_dataset.X = [1]
        stack.enter_context(
            patch(
                "brain_api.routes.training.snapshot_phase.build_dataset",
                return_value=mock_dataset,
            )
        )
        mock_result = MagicMock()
        mock_result.train_loss = 0.01
        mock_result.val_loss = 0.02
        stack.enter_context(
            patch(
                "brain_api.routes.training.snapshot_phase.train_model_pytorch",
                return_value=mock_result,
            )
        )
        stack.enter_context(patch("brain_api.routes.training.snapshot_phase.gc"))
        stack.enter_context(patch("brain_api.routes.training.snapshot_phase.torch"))
        return stack

    def test_lstm_backfill_local_first_no_repo_skips_hf_check(self):
        from brain_api.routes.training.snapshot_phase import (
            _backfill_lstm_snapshots,
        )
        from brain_api.storage.policy import StoragePolicy

        mock_storage = MagicMock(spec=SnapshotLocalStorage)
        mock_storage.snapshot_exists_anywhere.return_value = False
        mock_storage.forecaster_type = "lstm_halal_new"
        mock_storage._get_hf_repo.return_value = None

        with self._common_patches(mock_storage):
            _backfill_lstm_snapshots(
                symbols=["AAPL"],
                config=MagicMock(to_dict=dict),
                start_date=date(2016, 1, 1),
                end_date=date(2025, 12, 26),
                snapshot_storage=mock_storage,
                policy=StoragePolicy.LOCAL_FIRST,
            )

        for call in mock_storage.snapshot_exists_anywhere.call_args_list:
            assert call.kwargs["check_hf"] is False

    def test_lstm_backfill_hf_first_with_repo_consults_hf(self):
        from brain_api.routes.training.snapshot_phase import (
            _backfill_lstm_snapshots,
        )
        from brain_api.storage.policy import StoragePolicy

        mock_storage = MagicMock(spec=SnapshotLocalStorage)
        mock_storage.snapshot_exists_anywhere.return_value = True  # all present
        mock_storage.forecaster_type = "lstm_halal_new"
        mock_storage._get_hf_repo.return_value = "user/repo"

        with self._common_patches(mock_storage):
            _backfill_lstm_snapshots(
                symbols=["AAPL"],
                config=MagicMock(to_dict=dict),
                start_date=date(2016, 1, 1),
                end_date=date(2025, 12, 26),
                snapshot_storage=mock_storage,
                policy=StoragePolicy.HF_FIRST,
            )

        for call in mock_storage.snapshot_exists_anywhere.call_args_list:
            assert call.kwargs["check_hf"] is True

    def test_lstm_backfill_hf_first_no_repo_raises(self):
        from brain_api.routes.training.snapshot_phase import (
            _backfill_lstm_snapshots,
        )
        from brain_api.storage.policy import StoragePolicy, StoragePolicyError

        mock_storage = MagicMock(spec=SnapshotLocalStorage)
        mock_storage.forecaster_type = "lstm_halal_new"
        mock_storage._get_hf_repo.return_value = None

        with pytest.raises(StoragePolicyError):
            _backfill_lstm_snapshots(
                symbols=["AAPL"],
                config=MagicMock(to_dict=dict),
                start_date=date(2016, 1, 1),
                end_date=date(2025, 12, 26),
                snapshot_storage=mock_storage,
                policy=StoragePolicy.HF_FIRST,
            )

    def test_patchtst_backfill_hf_first_no_repo_raises(self):
        from brain_api.routes.training.snapshot_phase import (
            _backfill_patchtst_snapshots,
        )
        from brain_api.storage.policy import StoragePolicy, StoragePolicyError

        mock_storage = MagicMock(spec=SnapshotLocalStorage)
        mock_storage.forecaster_type = "patchtst_halal_new"
        mock_storage._get_hf_repo.return_value = None

        with pytest.raises(StoragePolicyError):
            _backfill_patchtst_snapshots(
                symbols=["AAPL"],
                config=MagicMock(to_dict=dict),
                start_date=date(2016, 1, 1),
                end_date=date(2025, 12, 26),
                snapshot_storage=mock_storage,
                policy=StoragePolicy.HF_FIRST,
            )

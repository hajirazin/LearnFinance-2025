"""Tests for forecaster snapshot training, storage, and walk-forward inference."""

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from brain_api.core.portfolio_rl.walkforward import (
    build_forecast_features,
    compute_momentum_proxy,
    generate_walkforward_forecasts_simple,
)
from brain_api.storage.forecaster_snapshots import (
    LSTMSnapshotArtifacts,
    PatchTSTSnapshotArtifacts,
    SnapshotLocalStorage,
    create_snapshot_metadata,
)


class TestSnapshotLocalStorage:
    """Tests for SnapshotLocalStorage class."""

    def test_init_creates_storage(self, tmp_path):
        """Test storage initialization."""
        storage = SnapshotLocalStorage("lstm", base_path=tmp_path)
        assert storage.forecaster_type == "lstm"
        assert storage.base_path == tmp_path

    def test_snapshot_path(self, tmp_path):
        """Test snapshot path generation (flat structure, sibling to main versions)."""
        storage = SnapshotLocalStorage("lstm", base_path=tmp_path)
        cutoff = date(2019, 12, 31)
        # New flat structure: models/lstm/snapshot-2019-12-31 (not in snapshots/ subfolder)
        expected = tmp_path / "models" / "lstm" / "snapshot-2019-12-31"
        assert storage._snapshot_path(cutoff) == expected

    def test_snapshot_exists_false(self, tmp_path):
        """Test snapshot_exists returns False when no snapshot."""
        storage = SnapshotLocalStorage("lstm", base_path=tmp_path)
        assert storage.snapshot_exists(date(2019, 12, 31)) is False

    def test_list_snapshots_empty(self, tmp_path):
        """Test list_snapshots returns empty when no snapshots."""
        storage = SnapshotLocalStorage("lstm", base_path=tmp_path)
        assert storage.list_snapshots() == []

    def test_write_and_list_snapshot(self, tmp_path):
        """Test writing and listing a snapshot."""
        storage = SnapshotLocalStorage("lstm", base_path=tmp_path)
        cutoff = date(2019, 12, 31)

        # Create mock model and scaler
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {"weight": torch.tensor([1.0])}
        mock_scaler = StandardScaler()
        mock_scaler.mean_ = np.array([0.0])
        mock_scaler.scale_ = np.array([1.0])
        mock_config = MagicMock()
        mock_config.to_dict.return_value = {"hidden_size": 64}
        metadata = {"test": "value"}

        # Write snapshot
        snapshot_path = storage.write_snapshot(
            cutoff_date=cutoff,
            model=mock_model,
            feature_scaler=mock_scaler,
            config=mock_config,
            metadata=metadata,
        )

        # Verify files exist
        assert (snapshot_path / "weights.pt").exists()
        assert (snapshot_path / "feature_scaler.pkl").exists()
        assert (snapshot_path / "config.json").exists()
        assert (snapshot_path / "metadata.json").exists()

        # List snapshots
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
            model=mock_model,
            feature_scaler=StandardScaler(),
            config=mock_config,
            metadata={},
        )

        assert storage.snapshot_exists(cutoff) is True

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
                cutoff, mock_model, StandardScaler(), mock_config, {}
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
        storage.write_snapshot(cutoff, mock_model, StandardScaler(), mock_config, {})

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
        storage.write_snapshot(cutoff, mock_model, StandardScaler(), mock_config, {})

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
            cutoff, mock_model, StandardScaler(), mock_config, metadata
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
        )

        assert metadata["forecaster_type"] == "lstm"
        assert metadata["cutoff_date"] == "2019-12-31"
        assert metadata["data_window"]["start"] == "2016-01-01"
        assert metadata["data_window"]["end"] == "2019-12-31"
        assert metadata["symbols"] == ["AAPL", "MSFT"]
        assert metadata["metrics"]["train_loss"] == 0.01
        assert metadata["metrics"]["val_loss"] == 0.02
        assert "training_timestamp" in metadata


class TestWalkForwardForecasts:
    """Tests for walk-forward forecast generation."""

    def test_compute_momentum_proxy(self):
        """Test momentum proxy computation."""
        prices = np.array([100, 102, 104, 106, 108, 110, 112])
        momentum = compute_momentum_proxy(prices, lookback_weeks=4)

        # First 4 values should be 0 (not enough history)
        assert all(momentum[:4] == 0)
        # After that, momentum should be computed
        assert momentum[4] > 0  # 108 vs 100
        assert momentum[5] > 0  # 110 vs 102

    def test_compute_momentum_proxy_short_array(self):
        """Test momentum with short price array."""
        prices = np.array([100, 102, 104])
        momentum = compute_momentum_proxy(prices, lookback_weeks=4)
        assert len(momentum) == 3
        assert all(momentum == 0)

    def test_generate_simple_forecasts(self):
        """Test simple momentum-based forecast generation."""
        # Create weekly prices (2 years of data)
        n_weeks = 104  # ~2 years
        weekly_prices = {
            "AAPL": np.linspace(100, 200, n_weeks),  # Uptrend
            "MSFT": np.linspace(150, 120, n_weeks),  # Downtrend
        }
        weekly_dates = pd.date_range("2020-01-06", periods=n_weeks, freq="W-MON")
        symbols = ["AAPL", "MSFT"]

        forecasts = generate_walkforward_forecasts_simple(
            weekly_prices, weekly_dates, symbols, bootstrap_years=1
        )

        assert "AAPL" in forecasts
        assert "MSFT" in forecasts
        assert len(forecasts["AAPL"]) == n_weeks - 1

    def test_generate_simple_forecasts_bootstrap(self):
        """Test bootstrap period uses zeros."""
        n_weeks = 52  # 1 year
        weekly_prices = {
            "AAPL": np.linspace(100, 150, n_weeks),
        }
        weekly_dates = pd.date_range("2020-01-06", periods=n_weeks, freq="W-MON")

        # With 4 year bootstrap, all of 2020 should be zeros
        forecasts = generate_walkforward_forecasts_simple(
            weekly_prices, weekly_dates, ["AAPL"], bootstrap_years=4
        )

        # All forecasts should be 0 since we're in bootstrap period
        assert all(forecasts["AAPL"] == 0)

    def test_build_forecast_features_simple(self):
        """Test build_forecast_features with momentum proxy."""
        n_weeks = 52
        weekly_prices = {
            "AAPL": np.linspace(100, 150, n_weeks),
        }
        weekly_dates = pd.date_range("2020-01-06", periods=n_weeks, freq="W-MON")

        forecasts = build_forecast_features(
            weekly_prices,
            weekly_dates,
            ["AAPL"],
            forecaster_type="lstm",
            use_model_snapshots=False,
        )

        assert "AAPL" in forecasts
        assert len(forecasts["AAPL"]) == n_weeks - 1

    def test_build_forecast_features_with_missing_snapshots(self, tmp_path):
        """Test build_forecast_features falls back when no snapshots."""
        n_weeks = 52
        weekly_prices = {
            "AAPL": np.linspace(100, 150, n_weeks),
        }
        weekly_dates = pd.date_range("2020-01-06", periods=n_weeks, freq="W-MON")

        # Use model snapshots but they don't exist
        with patch(
            "brain_api.core.portfolio_rl.walkforward.generate_walkforward_forecasts_with_model"
        ) as mock:
            mock.return_value = generate_walkforward_forecasts_simple(
                weekly_prices, weekly_dates, ["AAPL"]
            )
            forecasts = build_forecast_features(
                weekly_prices,
                weekly_dates,
                ["AAPL"],
                forecaster_type="lstm",
                use_model_snapshots=True,
            )

        assert "AAPL" in forecasts

    def test_generate_forecasts_empty_prices(self):
        """Test forecast generation with empty prices."""
        forecasts = generate_walkforward_forecasts_simple({}, pd.DatetimeIndex([]), [])
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


class TestPatchTSTSnapshots:
    """Tests specific to PatchTST snapshot handling."""

    def test_patchtst_storage_type(self, tmp_path):
        """Test PatchTST storage is properly typed."""
        storage = SnapshotLocalStorage("patchtst", base_path=tmp_path)
        assert storage.forecaster_type == "patchtst"
        # New flat structure: models path where snapshots live as siblings to main versions
        expected = tmp_path / "models" / "patchtst"
        assert storage._models_path == expected

    def test_patchtst_snapshot_path(self, tmp_path):
        """Test PatchTST snapshot path generation (flat structure)."""
        storage = SnapshotLocalStorage("patchtst", base_path=tmp_path)
        cutoff = date(2019, 12, 31)
        # New flat structure: models/patchtst/snapshot-2019-12-31 (not in snapshots/ subfolder)
        expected = tmp_path / "models" / "patchtst" / "snapshot-2019-12-31"
        assert storage._snapshot_path(cutoff) == expected


class TestSnapshotInferenceHelpers:
    """Tests for snapshot inference helper functions."""

    def test_lstm_inference_fallback_short_history(self):
        """Test LSTM inference falls back with short history."""
        from brain_api.core.portfolio_rl.walkforward import _run_lstm_snapshot_inference

        # Create mock artifacts
        mock_config = MagicMock()
        mock_config.sequence_length = 20
        mock_config.use_returns = True

        mock_model = MagicMock()
        mock_scaler = StandardScaler()

        artifacts = LSTMSnapshotArtifacts(
            config=mock_config,
            feature_scaler=mock_scaler,
            model=mock_model,
            cutoff_date=date(2019, 12, 31),
        )

        prices = np.linspace(100, 110, 10)  # Only 10 weeks
        year_indices = [5, 6, 7]  # Indices to predict

        # Create weekly dates for the test
        weekly_dates = pd.date_range(start="2019-01-01", periods=10, freq="W-FRI")

        # Should fall back to momentum since no daily OHLCV available
        predictions = _run_lstm_snapshot_inference(
            artifacts,
            prices,
            year_indices,
            weekly_dates=weekly_dates,
            symbol="TEST",
        )
        assert len(predictions) == 3

    def test_patchtst_inference_fallback_short_history(self):
        """Test PatchTST inference falls back with short history."""
        from brain_api.core.portfolio_rl.walkforward import (
            _run_patchtst_snapshot_inference,
        )

        # Create mock artifacts
        mock_config = MagicMock()
        mock_config.context_length = 20

        mock_model = MagicMock()
        mock_scaler = StandardScaler()

        artifacts = PatchTSTSnapshotArtifacts(
            config=mock_config,
            feature_scaler=mock_scaler,
            model=mock_model,
            cutoff_date=date(2019, 12, 31),
        )

        prices = np.linspace(100, 110, 10)  # Only 10 weeks
        year_indices = [5, 6, 7]  # Indices to predict

        # Should fall back to momentum since context_length=20 > len(prices)
        predictions = _run_patchtst_snapshot_inference(artifacts, prices, year_indices)
        assert len(predictions) == 3

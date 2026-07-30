"""Tests for snapshot walk-forward forecasts and inference helpers.

Covers:

* :class:`TestWalkForwardForecasts` -- ``build_forecast_features`` +
  ``generate_walkforward_forecasts`` behaviour, including the
  must-have ``SnapshotUnavailableError`` raise when no snapshot is
  reachable.
* :class:`TestSnapshotIntegration` -- the ``_snapshots_available``
  helper used by the training endpoints, plus a smoke test that the
  re-exported backfill functions are callable.
* :class:`TestSnapshotInferenceHelpers` -- the ``_run_*_snapshot_inference``
  helpers' guard against missing ``weekly_dates``.
"""

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from brain_api.core.portfolio_rl.walkforward import build_forecast_features
from brain_api.storage.forecaster_snapshots import (
    LSTMSnapshotArtifacts,
    PatchTSTSnapshotArtifacts,
    SnapshotLocalStorage,
)


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

    def test_lstm_next_week_snapshot_includes_cutoff_friday(self, monkeypatch):
        """Friday close is known before the following Monday-open action."""
        import numpy as np
        import pandas as pd
        import torch

        from brain_api.core.portfolio_rl.walkforward import (
            _predict_single_week_lstm,
        )

        friday = pd.Timestamp("2026-01-09")
        daily = pd.DataFrame(
            {
                "open": [1.0, 2.0, 3.0],
                "high": [1.0, 2.0, 3.0],
                "low": [1.0, 2.0, 3.0],
                "close": [1.0, 2.0, 3.0],
                "volume": [1.0, 2.0, 3.0],
            },
            index=pd.DatetimeIndex(["2026-01-07", "2026-01-08", friday]),
        )
        captured: dict[str, torch.Tensor] = {}

        class Model:
            def __call__(self, values):
                captured["values"] = values
                return torch.zeros((1, 5))

        def identity_features(frame, use_returns):
            del use_returns
            return frame

        monkeypatch.setattr(
            "brain_api.core.features.compute_ohlcv_log_returns",
            identity_features,
        )
        result = _predict_single_week_lstm(
            model=Model(),
            scaler=None,
            config=type(
                "Config",
                (),
                {"sequence_length": 2, "use_returns": True},
            )(),
            weekly_idx=0,
            weekly_dates=pd.DatetimeIndex([friday]),
            daily_ohlcv=daily,
            symbol="AAA",
        )

        assert result == pytest.approx(0.0)
        assert np.asarray(captured["values"])[0, -1, 3] == pytest.approx(3.0)

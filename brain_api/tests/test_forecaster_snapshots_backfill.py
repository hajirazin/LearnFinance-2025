"""Tests for the snapshot backfill loops + cutoff filter helpers.

Covers:

* :class:`TestBackfillSnapshotRange` -- the LSTM and PatchTST backfill
  loops produce the right cutoffs (start_year-1 .. end_year-1), skip
  pre-existing snapshots, and stamp the extended ``data_window_start``
  on the metadata.
* :class:`TestFilterByCutoffTzHandling` -- regression coverage for the
  tz-aware ``DatetimeIndex`` bug; the snapshot-phase filter helpers
  must not crash on yfinance frames whose index carries a timezone.
* :class:`TestBackfillLoopsRespectPolicy` -- the ``policy``-aware
  refactor of the backfill loops (no silent ``check_hf = repo is not
  None`` collapse; ``hf_first`` + missing repo MUST raise).
"""

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from brain_api.storage.forecaster_snapshots import SnapshotLocalStorage


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


class TestFilterByCutoffTzHandling:
    """Regression tests for the snapshot-phase filter helpers.

    Real yfinance loads return ``DatetimeIndex(tz="America/New_York")``;
    the backfill cutoff is a naive ``date``. Direct comparison raises
    ``Invalid comparison between dtype=datetime64[ns, America/New_York]
    and Timestamp``. These tests pin the localize-on-the-fly behaviour
    that mirrors :mod:`brain_api.core.lstm.inference` lines 89-91 and
    prevents the snapshots-only background runner from crashing on a
    real backfill.
    """

    def test_filter_prices_by_cutoff_handles_tz_aware_index(self):
        """Tz-aware index + tz-naive cutoff must NOT raise. Filter
        keeps only rows on/before the cutoff date (in the index's tz)."""
        from brain_api.routes.training.snapshot_phase import (
            _filter_prices_by_cutoff,
        )

        index = pd.date_range(
            "2023-01-01", "2024-06-30", freq="B", tz="America/New_York"
        )
        df = pd.DataFrame({"close": np.arange(len(index), dtype=float)}, index=index)
        prices = {"AAPL": df}

        out = _filter_prices_by_cutoff(prices, date(2023, 12, 31))

        assert "AAPL" in out
        # Last kept row's date in New_York must be <= cutoff.
        assert out["AAPL"].index[-1].date() <= date(2023, 12, 31)
        assert out["AAPL"].index[0].date() >= date(2023, 1, 1)
        # Original frame untouched (the helper returns a copy).
        assert len(prices["AAPL"]) == len(index)

    def test_filter_prices_by_cutoff_drops_symbols_with_no_rows_after_cutoff(self):
        """Symbol whose entire tz-aware index post-dates the cutoff is
        dropped, just like the legacy tz-naive code path."""
        from brain_api.routes.training.snapshot_phase import (
            _filter_prices_by_cutoff,
        )

        future_index = pd.date_range(
            "2025-01-01", "2025-06-30", freq="B", tz="America/New_York"
        )
        df = pd.DataFrame(
            {"close": np.arange(len(future_index), dtype=float)},
            index=future_index,
        )
        prices = {"AAPL": df}

        out = _filter_prices_by_cutoff(prices, date(2020, 12, 31))

        assert out == {}

    def test_filter_prices_by_cutoff_preserves_tz_naive_behaviour(self):
        """Tz-naive callers (the existing test suite) must see no change.

        Locks in that the localize branch only fires when ``df.index.tz``
        is set; tz-naive frames go through the original comparison.
        """
        from brain_api.routes.training.snapshot_phase import (
            _filter_prices_by_cutoff,
        )

        index = pd.date_range("2023-01-01", "2024-06-30", freq="B")
        df = pd.DataFrame({"close": np.arange(len(index), dtype=float)}, index=index)
        prices = {"AAPL": df}

        out = _filter_prices_by_cutoff(prices, date(2023, 12, 31))

        assert "AAPL" in out
        assert out["AAPL"].index.tz is None
        assert out["AAPL"].index[-1].date() <= date(2023, 12, 31)

    def test_filter_signals_by_cutoff_handles_tz_aware_index(self):
        """Same tz-aware regression for the signals helper. The signals
        path is currently unused by ``_backfill_patchtst_snapshots``
        but the helper signature still accepts arbitrary indexes; this
        pins the localize fix so a future caller cannot regress it.
        """
        from brain_api.routes.training.snapshot_phase import (
            _filter_signals_by_cutoff,
        )

        index = pd.date_range(
            "2023-01-01", "2024-06-30", freq="B", tz="America/New_York"
        )
        df = pd.DataFrame({"sentiment": np.linspace(-1, 1, len(index))}, index=index)
        signals = {"AAPL": df}

        out = _filter_signals_by_cutoff(signals, date(2023, 12, 31))

        assert "AAPL" in out
        assert out["AAPL"].index[-1].date() <= date(2023, 12, 31)


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

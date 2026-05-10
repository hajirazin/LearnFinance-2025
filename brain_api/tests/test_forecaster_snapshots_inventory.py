"""Tests for the read-side snapshot inventory + ``check_hf`` resolver.

These helpers are the read-side mirror of the backfill loops -- they
must agree bit-for-bit with the trainer code on which cutoffs and
which digests get inspected. A drift here silently corrupts every
downstream snapshot decision (AGENTS.md rule #2).

Covers:

* :class:`TestResolveCheckHF` -- the 4-row truth table for the
  ``StoragePolicy`` -> ``check_hf`` translator. ``hf_first`` + no
  HF repo MUST raise (no silent fallback to local).
* :class:`TestCountMissingSnapshots` -- the inventory counter:
  cutoff/digest math matches the backfill formula, ``policy`` is
  threaded into every existence check, and the empty / partial /
  full-miss return shapes are pinned.
"""

from datetime import date
from unittest.mock import MagicMock

import pytest

from brain_api.storage.forecaster_snapshots import SnapshotLocalStorage


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

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

    SAC snapshot availability still uses this translator. Annual
    inventory and backfill call ``annual_snapshot_parity`` and do not.
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


class TestAnnualSnapshotCutoffs:
    def test_october_window_stops_at_prior_december(self):
        from brain_api.core.forecaster_snapshot_identity import (
            annual_snapshot_cutoffs,
        )

        cutoffs = annual_snapshot_cutoffs(date(2016, 1, 1), date(2026, 10, 2))

        assert date(2026, 10, 2) not in cutoffs
        assert date(2026, 12, 31) not in cutoffs
        assert cutoffs[0] == date(2015, 12, 31)
        assert cutoffs[-1] == date(2025, 12, 31)

    def test_december_31_window_includes_that_date_once(self):
        from brain_api.core.forecaster_snapshot_identity import (
            annual_snapshot_cutoffs,
        )

        cutoffs = annual_snapshot_cutoffs(date(2016, 1, 1), date(2025, 12, 31))

        assert cutoffs[-1] == date(2025, 12, 31)
        assert cutoffs.count(date(2025, 12, 31)) == 1


class TestAnnualSnapshotParity:
    @staticmethod
    def _storage(*, local: bool, repo: str | None, on_hf: bool) -> MagicMock:
        storage = MagicMock(spec=SnapshotLocalStorage)
        storage.snapshot_exists.return_value = local
        storage._get_hf_repo.return_value = repo
        storage.snapshot_digest_exists_on_hf.return_value = on_hf
        return storage

    def test_both_sides_ready(self):
        from brain_api.core.forecaster_snapshot_identity import (
            annual_snapshot_parity,
        )

        storage = self._storage(local=True, repo="user/repo", on_hf=True)
        assert annual_snapshot_parity(storage, date(2024, 12, 31), "abc") == "ready"

    def test_local_only_upload(self):
        from brain_api.core.forecaster_snapshot_identity import (
            annual_snapshot_parity,
        )

        storage = self._storage(local=True, repo="user/repo", on_hf=False)
        assert annual_snapshot_parity(storage, date(2024, 12, 31), "abc") == "upload"

    def test_hugging_face_only_download(self):
        from brain_api.core.forecaster_snapshot_identity import (
            annual_snapshot_parity,
        )

        storage = self._storage(local=False, repo="user/repo", on_hf=True)
        assert annual_snapshot_parity(storage, date(2024, 12, 31), "abc") == "download"

    def test_neither_side_trains(self):
        from brain_api.core.forecaster_snapshot_identity import (
            annual_snapshot_parity,
        )

        storage = self._storage(local=False, repo="user/repo", on_hf=False)
        assert annual_snapshot_parity(storage, date(2024, 12, 31), "abc") == "train"

    def test_rejected_only_trains(self):
        """A rejected directory is not ``snapshot_exists``, so parity trains."""
        from brain_api.core.forecaster_snapshot_identity import (
            annual_snapshot_parity,
        )

        storage = self._storage(local=False, repo="user/repo", on_hf=False)
        assert annual_snapshot_parity(storage, date(2024, 12, 31), "abc") == "train"

    def test_different_digest_trains(self):
        from brain_api.core.forecaster_snapshot_identity import (
            annual_snapshot_parity,
        )

        storage = self._storage(local=False, repo="user/repo", on_hf=False)
        assert (
            annual_snapshot_parity(storage, date(2024, 12, 31), "expected") == "train"
        )

    @pytest.mark.parametrize("backend", ["local_first", "hf_first"])
    def test_storage_backend_does_not_change_the_label(self, monkeypatch, backend):
        from brain_api.core.forecaster_snapshot_identity import (
            annual_snapshot_parity,
        )

        monkeypatch.setenv("STORAGE_BACKEND", backend)
        storage = self._storage(local=True, repo="user/repo", on_hf=False)
        assert annual_snapshot_parity(storage, date(2024, 12, 31), "abc") == "upload"

    def test_hf_first_without_repo_does_not_raise(self, monkeypatch):
        from brain_api.core.forecaster_snapshot_identity import (
            annual_snapshot_parity,
        )

        monkeypatch.setenv("STORAGE_BACKEND", "hf_first")
        local_only = self._storage(local=True, repo=None, on_hf=False)
        missing = self._storage(local=False, repo=None, on_hf=False)
        assert annual_snapshot_parity(local_only, date(2024, 12, 31), "abc") == "ready"
        assert annual_snapshot_parity(missing, date(2024, 12, 31), "abc") == "train"
        missing.snapshot_digest_exists_on_hf.assert_not_called()


class TestCountMissingSnapshots:
    """Read-side mirror of the annual backfill loops."""

    @staticmethod
    def _annual_digests(
        forecaster_type: str,
        train_window: tuple[date, date],
        config_dict: dict,
    ) -> list[tuple[date, str]]:
        from brain_api.core.forecaster_snapshot_identity import (
            annual_snapshot_cutoffs,
        )
        from brain_api.core.version import compute_snapshot_identity_hash

        start_date, end_date = train_window
        return [
            (
                cutoff,
                compute_snapshot_identity_hash(forecaster_type, cutoff, config_dict),
            )
            for cutoff in annual_snapshot_cutoffs(start_date, end_date)
        ]

    def _build_storage(self, *, hf_repo: str | None, exists_map: dict) -> MagicMock:
        storage = MagicMock(spec=SnapshotLocalStorage)
        storage.forecaster_type = "lstm_halal_new"
        storage._get_hf_repo.return_value = hf_repo

        def local_side_effect(cutoff, digest):
            return exists_map.get((cutoff, digest), False)

        storage.snapshot_exists.side_effect = local_side_effect
        storage.snapshot_digest_exists_on_hf.side_effect = local_side_effect
        return storage

    def test_ready_annual_cutoffs_ignore_the_raw_end_date(self):
        from brain_api.core.forecaster_snapshot_identity import (
            count_missing_snapshots,
        )

        train_window = (date(2016, 1, 1), date(2026, 10, 2))
        config_dict = {"k": "v"}
        annual = self._annual_digests("lstm_halal_new", train_window, config_dict)
        exists_map = dict.fromkeys(annual, True)
        storage = self._build_storage(hf_repo=None, exists_map=exists_map)

        inventory = count_missing_snapshots(
            forecaster_type="lstm_halal_new",
            train_window=train_window,
            config_dict=config_dict,
            snapshot_storage=storage,
        )
        assert inventory.is_empty
        assert inventory.total_missing == 0
        assert not hasattr(inventory, "end_window_cutoff")
        assert date(2026, 10, 2) not in inventory.historical_cutoffs

    def test_one_annual_cutoff_missing(self):
        from brain_api.core.forecaster_snapshot_identity import (
            count_missing_snapshots,
        )

        train_window = (date(2016, 1, 1), date(2025, 12, 26))
        config_dict = {"k": "v"}
        annual = self._annual_digests("lstm_halal_new", train_window, config_dict)
        exists_map = dict.fromkeys(annual[1:], True)
        storage = self._build_storage(hf_repo=None, exists_map=exists_map)

        inventory = count_missing_snapshots(
            forecaster_type="lstm_halal_new",
            train_window=train_window,
            config_dict=config_dict,
            snapshot_storage=storage,
        )
        assert inventory.historical_cutoffs == (annual[0][0],)
        assert inventory.total_missing == 1

    def test_two_annual_cutoffs_missing(self):
        from brain_api.core.forecaster_snapshot_identity import (
            count_missing_snapshots,
        )

        train_window = (date(2016, 1, 1), date(2025, 12, 26))
        config_dict = {"k": "v"}
        annual = self._annual_digests("lstm_halal_new", train_window, config_dict)
        exists_map = dict.fromkeys(annual[2:], True)
        storage = self._build_storage(hf_repo=None, exists_map=exists_map)

        inventory = count_missing_snapshots(
            forecaster_type="lstm_halal_new",
            train_window=train_window,
            config_dict=config_dict,
            snapshot_storage=storage,
        )
        assert inventory.historical_cutoffs == (annual[0][0], annual[1][0])
        assert inventory.total_missing == 2

    def test_all_missing(self):
        from brain_api.core.forecaster_snapshot_identity import (
            count_missing_snapshots,
        )

        train_window = (date(2016, 1, 1), date(2025, 12, 26))
        config_dict = {"k": "v"}
        annual = self._annual_digests("lstm_halal_new", train_window, config_dict)
        storage = self._build_storage(hf_repo=None, exists_map={})

        inventory = count_missing_snapshots(
            forecaster_type="lstm_halal_new",
            train_window=train_window,
            config_dict=config_dict,
            snapshot_storage=storage,
        )
        assert inventory.historical_cutoffs == tuple(c for c, _ in annual)
        assert inventory.total_missing == len(annual)

    def test_both_policies_inventory_the_same_cutoffs(self):
        from brain_api.core.forecaster_snapshot_identity import (
            count_missing_snapshots,
        )
        from brain_api.storage.policy import StoragePolicy

        train_window = (date(2016, 1, 1), date(2025, 12, 26))
        config_dict = {"k": "v"}
        totals = []
        for policy in (StoragePolicy.LOCAL_FIRST, StoragePolicy.HF_FIRST):
            storage = self._build_storage(hf_repo="user/repo", exists_map={})
            inventory = count_missing_snapshots(
                forecaster_type="lstm_halal_new",
                train_window=train_window,
                config_dict=config_dict,
                snapshot_storage=storage,
                policy=policy,
            )
            totals.append(inventory.total_missing)
        assert totals[0] == totals[1]
        assert totals[0] > 0

    def test_hf_first_without_repo_does_not_raise(self):
        from brain_api.core.forecaster_snapshot_identity import (
            count_missing_snapshots,
        )
        from brain_api.storage.policy import StoragePolicy

        train_window = (date(2016, 1, 1), date(2025, 12, 26))
        storage = self._build_storage(hf_repo=None, exists_map={})

        inventory = count_missing_snapshots(
            forecaster_type="lstm_halal_new",
            train_window=train_window,
            config_dict={"k": "v"},
            snapshot_storage=storage,
            policy=StoragePolicy.HF_FIRST,
        )
        assert inventory.total_missing > 0
        storage.snapshot_digest_exists_on_hf.assert_not_called()

    def test_policy_argument_is_ignored(self, monkeypatch):
        from brain_api.core.forecaster_snapshot_identity import (
            count_missing_snapshots,
        )

        monkeypatch.setenv("STORAGE_BACKEND", "hf_first")
        train_window = (date(2016, 1, 1), date(2025, 12, 26))
        storage = self._build_storage(hf_repo=None, exists_map={})

        inventory = count_missing_snapshots(
            forecaster_type="lstm_halal_new",
            train_window=train_window,
            config_dict={"k": "v"},
            snapshot_storage=storage,
        )
        assert inventory.total_missing > 0
        storage.snapshot_digest_exists_on_hf.assert_not_called()

    def test_digest_inputs_match_snapshot_identity_formula(self):
        from brain_api.core.forecaster_snapshot_identity import (
            count_missing_snapshots,
        )

        train_window = (date(2016, 1, 1), date(2025, 12, 26))
        config_dict = {"hidden": 16}
        storage = self._build_storage(hf_repo=None, exists_map={})

        count_missing_snapshots(
            forecaster_type="lstm_halal_new",
            train_window=train_window,
            config_dict=config_dict,
            snapshot_storage=storage,
        )

        annual = self._annual_digests("lstm_halal_new", train_window, config_dict)
        observed = [
            (call.args[0], call.args[1])
            for call in storage.snapshot_exists.call_args_list
        ]
        assert observed == annual
        assert (train_window[1], annual[0][1]) not in observed


def test_walkforward_expectation_bundles_do_not_resolve_symbols(monkeypatch):
    from brain_api.core import forecaster_snapshot_identity as identity
    from brain_api.core.lstm.config import DEFAULT_CONFIG as LSTM_DEFAULT_CONFIG
    from brain_api.core.patchtst.config import (
        DEFAULT_CONFIG as PATCHTST_DEFAULT_CONFIG,
    )
    from brain_api.universe import halal_new

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("snapshot identity must not resolve a universe")

    # Trap both historical lookup routes: bucket resolution and a direct/lazy
    # import of the underlying universe resolver. The identity module no longer
    # exposes either name, so the compatibility traps also catch a regression
    # that reintroduces an imported alias.
    monkeypatch.setattr(identity, "get_bucket", fail_if_called, raising=False)
    monkeypatch.setattr(
        identity, "get_halal_new_symbols", fail_if_called, raising=False
    )
    monkeypatch.setattr(halal_new, "get_halal_new_symbols", fail_if_called)

    assert identity.lstm_walkforward_expectation_bundle() == (
        "lstm_halal_new",
        LSTM_DEFAULT_CONFIG.to_dict(),
    )
    assert identity.patchtst_walkforward_expectation_bundle() == (
        "patchtst_halal_new",
        PATCHTST_DEFAULT_CONFIG.to_dict(),
    )


def test_expected_dec31_hash_uses_bucket_cutoff_and_config() -> None:
    from brain_api.core.forecaster_snapshot_identity import (
        expected_dec31_walkforward_snapshot_hash,
    )
    from brain_api.core.version import compute_snapshot_identity_hash

    cutoff = date(2020, 12, 31)
    config = {"hidden": 16}

    assert expected_dec31_walkforward_snapshot_hash(
        forecaster_bucket="lstm_halal_new",
        cutoff_date=cutoff,
        config_dict=config,
    ) == compute_snapshot_identity_hash("lstm_halal_new", cutoff, config)

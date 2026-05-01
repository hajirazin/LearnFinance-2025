"""Tests for HuggingFace storage and HF-aware policy helpers."""

from datetime import date
from unittest.mock import MagicMock, patch

import pytest
from sklearn.preprocessing import StandardScaler

from brain_api.storage.forecaster_snapshots import SnapshotLocalStorage
from brain_api.storage.policy import (
    StoragePolicy,
    StoragePolicyError,
    get_prior_metadata_for_bucket,
    hf_versions_match,
)


def _make_bucket(
    *,
    local_class: type | MagicMock,
    hf_class: type | MagicMock,
    hf_repo: str | None,
    bucket_name: str = "test_bucket",
) -> MagicMock:
    """Build a minimal mock BucketConfig for the policy helper tests.

    The helper only touches ``local_storage_class``, ``hf_storage_class``,
    ``hf_repo_getter``, and ``bucket_name``; the rest of ``BucketConfig``
    is irrelevant here, so a mock is plenty.
    """
    bucket = MagicMock()
    bucket.local_storage_class = local_class
    bucket.hf_storage_class = hf_class
    bucket.hf_repo_getter = lambda: hf_repo
    bucket.bucket_name = bucket_name
    return bucket


class TestGetPriorMetadataForBucketLocalFirst:
    """``local_first``: prefer local metadata; fall back to HF main."""

    def test_returns_local_metadata_when_present(self) -> None:
        local_storage = MagicMock()
        local_storage.read_current_version.return_value = "v2025-01-01-abc"
        local_storage.read_metadata.return_value = {
            "version": "v2025-01-01-abc",
            "metrics": {"val_loss": 0.05},
        }
        bucket = _make_bucket(
            local_class=lambda: local_storage,
            hf_class=MagicMock(),
            hf_repo=None,
        )

        meta = get_prior_metadata_for_bucket(
            bucket=bucket, policy=StoragePolicy.LOCAL_FIRST
        )

        assert meta == {
            "version": "v2025-01-01-abc",
            "metrics": {"val_loss": 0.05},
        }

    def test_returns_none_when_local_empty_and_no_hf_repo(self) -> None:
        local_storage = MagicMock()
        local_storage.read_current_version.return_value = None
        bucket = _make_bucket(
            local_class=lambda: local_storage,
            hf_class=MagicMock(),
            hf_repo=None,
        )

        assert (
            get_prior_metadata_for_bucket(
                bucket=bucket, policy=StoragePolicy.LOCAL_FIRST
            )
            is None
        )

    def test_falls_back_to_hf_when_local_empty(self) -> None:
        local_storage = MagicMock()
        local_storage.read_current_version.return_value = None
        hf_storage = MagicMock()
        hf_storage.token = "tok"
        bucket = _make_bucket(
            local_class=lambda: local_storage,
            hf_class=lambda **kwargs: hf_storage,
            hf_repo="user/repo",
        )

        with patch(
            "brain_api.storage.policy._fetch_hf_main_metadata",
            return_value={"version": "v2025-02-01-hf", "metrics": {"val_loss": 0.04}},
        ):
            meta = get_prior_metadata_for_bucket(
                bucket=bucket, policy=StoragePolicy.LOCAL_FIRST
            )

        assert meta == {
            "version": "v2025-02-01-hf",
            "metrics": {"val_loss": 0.04},
        }

    def test_local_first_swallows_hf_unreachable_as_no_prior(self) -> None:
        """Under local_first, HF transport failures must not abort training."""
        local_storage = MagicMock()
        local_storage.read_current_version.return_value = None
        hf_storage = MagicMock()
        hf_storage.token = None
        bucket = _make_bucket(
            local_class=lambda: local_storage,
            hf_class=lambda **kwargs: hf_storage,
            hf_repo="user/repo",
        )

        with patch(
            "brain_api.storage.policy._fetch_hf_main_metadata",
            side_effect=StoragePolicyError("network down"),
        ):
            meta = get_prior_metadata_for_bucket(
                bucket=bucket, policy=StoragePolicy.LOCAL_FIRST
            )

        assert meta is None


class TestGetPriorMetadataForBucketHFFirst:
    """``hf_first``: HF main is the source of truth for prior metadata."""

    def test_requires_hf_repo(self) -> None:
        local_storage = MagicMock()
        bucket = _make_bucket(
            local_class=lambda: local_storage,
            hf_class=MagicMock(),
            hf_repo=None,
        )

        with pytest.raises(StoragePolicyError):
            get_prior_metadata_for_bucket(bucket=bucket, policy=StoragePolicy.HF_FIRST)

    def test_returns_hf_metadata_when_present(self) -> None:
        local_storage = MagicMock()
        hf_storage = MagicMock()
        hf_storage.token = "tok"
        bucket = _make_bucket(
            local_class=lambda: local_storage,
            hf_class=lambda **kwargs: hf_storage,
            hf_repo="user/repo",
        )
        hf_meta = {"version": "v2025-03-01-hf", "metrics": {"val_loss": 0.03}}

        with patch(
            "brain_api.storage.policy._fetch_hf_main_metadata",
            return_value=hf_meta,
        ):
            meta = get_prior_metadata_for_bucket(
                bucket=bucket, policy=StoragePolicy.HF_FIRST
            )

        assert meta == hf_meta

    def test_returns_none_for_inaugural_promotion_when_hf_main_missing(self) -> None:
        local_storage = MagicMock()
        hf_storage = MagicMock()
        hf_storage.token = "tok"
        bucket = _make_bucket(
            local_class=lambda: local_storage,
            hf_class=lambda **kwargs: hf_storage,
            hf_repo="user/repo",
        )

        with patch(
            "brain_api.storage.policy._fetch_hf_main_metadata",
            return_value=None,
        ):
            meta = get_prior_metadata_for_bucket(
                bucket=bucket, policy=StoragePolicy.HF_FIRST
            )

        assert meta is None

    def test_propagates_storage_policy_error_on_hf_unreachable(self) -> None:
        local_storage = MagicMock()
        hf_storage = MagicMock()
        hf_storage.token = "tok"
        bucket = _make_bucket(
            local_class=lambda: local_storage,
            hf_class=lambda **kwargs: hf_storage,
            hf_repo="user/repo",
        )

        with (
            patch(
                "brain_api.storage.policy._fetch_hf_main_metadata",
                side_effect=StoragePolicyError("HF down"),
            ),
            pytest.raises(StoragePolicyError),
        ):
            get_prior_metadata_for_bucket(bucket=bucket, policy=StoragePolicy.HF_FIRST)


class TestHFVersionsMatch:
    """``hf_first``'s cheap short-circuit."""

    def test_match(self) -> None:
        assert hf_versions_match("v1", {"version": "v1"}) is True

    def test_mismatch(self) -> None:
        assert hf_versions_match("v1", {"version": "v2"}) is False

    def test_none_local(self) -> None:
        assert hf_versions_match(None, {"version": "v1"}) is False

    def test_none_hf(self) -> None:
        assert hf_versions_match("v1", None) is False


class TestSnapshotExistsAnywhere:
    """``snapshot_exists_anywhere`` short-circuits on local hits."""

    def test_returns_true_when_local_exists(self, tmp_path):
        storage = SnapshotLocalStorage("lstm_halal_new", base_path=tmp_path)
        cutoff = date(2019, 12, 31)

        mock_model = MagicMock()
        mock_model.state_dict.return_value = {}
        mock_config = MagicMock()
        mock_config.to_dict.return_value = {}
        storage.write_snapshot(cutoff, mock_model, StandardScaler(), mock_config, {})

        assert storage.snapshot_exists_anywhere(cutoff, check_hf=False) is True
        assert storage.snapshot_exists_anywhere(cutoff, check_hf=True) is True

    def test_returns_false_when_no_local_and_check_hf_false(self, tmp_path):
        storage = SnapshotLocalStorage("lstm_halal_new", base_path=tmp_path)
        cutoff = date(2019, 12, 31)

        assert storage.snapshot_exists_anywhere(cutoff, check_hf=False) is False

    def test_checks_hf_when_no_local_and_check_hf_true(self, tmp_path):
        storage = SnapshotLocalStorage("lstm_halal_new", base_path=tmp_path)
        cutoff = date(2019, 12, 31)

        with patch.object(storage, "list_hf_snapshots", return_value=[cutoff]):
            assert storage.snapshot_exists_anywhere(cutoff, check_hf=True) is True

    def test_returns_false_when_not_in_local_or_hf(self, tmp_path):
        storage = SnapshotLocalStorage("lstm_halal_new", base_path=tmp_path)
        cutoff = date(2019, 12, 31)

        with patch.object(storage, "list_hf_snapshots", return_value=[]):
            assert storage.snapshot_exists_anywhere(cutoff, check_hf=True) is False

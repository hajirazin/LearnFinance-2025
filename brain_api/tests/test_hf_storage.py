"""Tests for HuggingFace storage and HF-aware helper functions."""

from datetime import date
from unittest.mock import MagicMock, patch

from sklearn.preprocessing import StandardScaler

from brain_api.routes.training.helpers import PriorVersionInfo, get_prior_version_info
from brain_api.storage.forecaster_snapshots import SnapshotLocalStorage


class TestGetPriorVersionInfo:
    """Tests for get_prior_version_info helper function."""

    def test_returns_local_version_when_exists(self, tmp_path):
        """Test that local version is returned when it exists."""
        # Create mock local storage
        mock_storage = MagicMock()
        mock_storage.read_current_version.return_value = "v2025-01-01-abc123"
        mock_storage.read_metadata.return_value = {
            "version": "v2025-01-01-abc123",
            "metrics": {"val_loss": 0.05},
        }

        result = get_prior_version_info(mock_storage)

        assert result.version == "v2025-01-01-abc123"
        assert result.val_loss == 0.05
        assert result.metadata is not None

    def test_returns_none_when_no_local_and_local_backend(self, tmp_path):
        """Test returns None values when no local version and storage backend is local."""
        mock_storage = MagicMock()
        mock_storage.read_current_version.return_value = None

        result = get_prior_version_info(mock_storage)

        assert result.version is None
        assert result.val_loss is None
        assert result.metadata is None

    def test_falls_back_to_hf_when_local_empty(self, tmp_path):
        """Test that HF is checked when local is empty and storage_backend=hf."""
        mock_storage = MagicMock()
        mock_storage.read_current_version.return_value = None

        mock_hf_storage_class = MagicMock()
        mock_hf_instance = MagicMock()
        mock_hf_instance.get_current_metadata.return_value = {
            "version": "v2025-01-01-hf123",
            "metrics": {"val_loss": 0.04},
        }
        mock_hf_storage_class.return_value = mock_hf_instance

        with patch(
            "brain_api.routes.training.helpers.get_storage_backend", return_value="hf"
        ):
            result = get_prior_version_info(
                local_storage=mock_storage,
                hf_storage_class=mock_hf_storage_class,
                hf_model_repo="test/repo",
            )

        assert result.version == "v2025-01-01-hf123"
        assert result.val_loss == 0.04
        assert result.metadata is not None

    def test_hf_fallback_handles_exception(self, tmp_path):
        """Test that HF fallback gracefully handles exceptions."""
        mock_storage = MagicMock()
        mock_storage.read_current_version.return_value = None

        mock_hf_storage_class = MagicMock()
        mock_hf_storage_class.side_effect = Exception("HF not available")

        with patch(
            "brain_api.routes.training.helpers.get_storage_backend", return_value="hf"
        ):
            result = get_prior_version_info(
                local_storage=mock_storage,
                hf_storage_class=mock_hf_storage_class,
                hf_model_repo="test/repo",
            )

        # Should return empty result, not raise
        assert result.version is None
        assert result.val_loss is None

    def test_no_hf_fallback_when_storage_backend_local(self, tmp_path):
        """Test that HF is not checked when storage_backend=local."""
        mock_storage = MagicMock()
        mock_storage.read_current_version.return_value = None

        mock_hf_storage_class = MagicMock()

        with patch(
            "brain_api.routes.training.helpers.get_storage_backend",
            return_value="local",
        ):
            result = get_prior_version_info(
                local_storage=mock_storage,
                hf_storage_class=mock_hf_storage_class,
                hf_model_repo="test/repo",
            )

        # HF should not be called
        mock_hf_storage_class.assert_not_called()
        assert result.version is None


class TestSnapshotExistsAnywhere:
    """Tests for snapshot_exists_anywhere method."""

    def test_returns_true_when_local_exists(self, tmp_path):
        """Test returns True when snapshot exists locally."""
        storage = SnapshotLocalStorage("lstm", base_path=tmp_path)
        cutoff = date(2019, 12, 31)

        # Create a snapshot locally
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {}
        mock_config = MagicMock()
        mock_config.to_dict.return_value = {}
        storage.write_snapshot(cutoff, mock_model, StandardScaler(), mock_config, {})

        # Should return True without checking HF
        assert storage.snapshot_exists_anywhere(cutoff, check_hf=False) is True
        assert storage.snapshot_exists_anywhere(cutoff, check_hf=True) is True

    def test_returns_false_when_no_local_and_check_hf_false(self, tmp_path):
        """Test returns False when no local snapshot and check_hf=False."""
        storage = SnapshotLocalStorage("lstm", base_path=tmp_path)
        cutoff = date(2019, 12, 31)

        assert storage.snapshot_exists_anywhere(cutoff, check_hf=False) is False

    def test_checks_hf_when_no_local_and_check_hf_true(self, tmp_path):
        """Test checks HF when no local snapshot and check_hf=True."""
        storage = SnapshotLocalStorage("lstm", base_path=tmp_path)
        cutoff = date(2019, 12, 31)

        # Mock list_hf_snapshots to return the cutoff date
        with patch.object(storage, "list_hf_snapshots", return_value=[cutoff]):
            assert storage.snapshot_exists_anywhere(cutoff, check_hf=True) is True

    def test_returns_false_when_not_in_local_or_hf(self, tmp_path):
        """Test returns False when snapshot not in local or HF."""
        storage = SnapshotLocalStorage("lstm", base_path=tmp_path)
        cutoff = date(2019, 12, 31)

        # Mock list_hf_snapshots to return empty list
        with patch.object(storage, "list_hf_snapshots", return_value=[]):
            assert storage.snapshot_exists_anywhere(cutoff, check_hf=True) is False


class TestPriorVersionInfoDataclass:
    """Tests for PriorVersionInfo dataclass."""

    def test_dataclass_creation(self):
        """Test PriorVersionInfo can be created with all fields."""
        info = PriorVersionInfo(
            version="v2025-01-01-abc123",
            metadata={"test": "value"},
            val_loss=0.05,
        )
        assert info.version == "v2025-01-01-abc123"
        assert info.metadata == {"test": "value"}
        assert info.val_loss == 0.05

    def test_dataclass_with_none_values(self):
        """Test PriorVersionInfo can be created with None values."""
        info = PriorVersionInfo(
            version=None,
            metadata=None,
            val_loss=None,
        )
        assert info.version is None
        assert info.metadata is None
        assert info.val_loss is None

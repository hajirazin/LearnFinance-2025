"""Tests for /models/active-symbols endpoint.

Covers the mandatory ``universe`` query param (no default) introduced
to support the parallel A/B SAC workflows. Per AGENTS.md rule #1
("no silent fallbacks") the endpoint must 422 when callers omit
``universe`` so the two SAC buckets cannot accidentally share state.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from brain_api.main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestActiveSymbolsUniverseRouting:
    """``/models/active-symbols`` requires ``universe`` and routes per bucket."""

    def test_missing_universe_returns_422(self, client):
        """No universe -> 422 (mandatory, no default)."""
        response = client.get("/models/active-symbols")
        assert response.status_code == 422
        body = response.json()
        assert any("universe" in str(loc) for loc in body.get("detail", []))

    def test_unknown_universe_returns_422(self, client):
        """Unknown universe -> 422 with the allow-list in the detail."""
        response = client.get(
            "/models/active-symbols", params={"universe": "halal_new"}
        )
        assert response.status_code == 422
        detail = response.json()["detail"]
        # Routes the unknown SAC universe to the bucket registry's
        # error message which lists allowed values for the SAC family.
        assert "halal_new" in detail
        assert "halal_filtered" in detail
        assert "halal" in detail

    def test_halal_filtered_routes_to_halal_filtered_storage(self, client):
        """`?universe=halal_filtered` reads from SACHalalFilteredModelStorage."""
        with patch("brain_api.routes.models_meta.get_bucket") as mock_get_bucket:
            fake_storage = MagicMock()
            fake_storage.read_current_version.return_value = "v2026-05-01-aaa"
            fake_storage.load_symbol_order.return_value = [f"S{i}" for i in range(15)]
            fake_bucket = MagicMock()
            fake_bucket.bucket_name = "sac_halal_filtered"
            fake_bucket.local_storage_class.return_value = fake_storage
            mock_get_bucket.return_value = fake_bucket

            response = client.get(
                "/models/active-symbols", params={"universe": "halal_filtered"}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["source_model"] == "sac_halal_filtered"
        assert data["model_version"] == "v2026-05-01-aaa"
        assert len(data["symbols"]) == 15

    def test_halal_routes_to_halal_storage(self, client):
        """`?universe=halal` reads from SACHalalModelStorage (variable size)."""
        with patch("brain_api.routes.models_meta.get_bucket") as mock_get_bucket:
            fake_storage = MagicMock()
            fake_storage.read_current_version.return_value = "v2026-05-01-bbb"
            # halal bucket is variable size (10-15 typical); use 14 to
            # exercise the n-agnostic path explicitly.
            fake_storage.load_symbol_order.return_value = [f"H{i}" for i in range(14)]
            fake_bucket = MagicMock()
            fake_bucket.bucket_name = "sac_halal"
            fake_bucket.local_storage_class.return_value = fake_storage
            mock_get_bucket.return_value = fake_bucket

            response = client.get(
                "/models/active-symbols", params={"universe": "halal"}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["source_model"] == "sac_halal"
        assert data["model_version"] == "v2026-05-01-bbb"
        assert len(data["symbols"]) == 14

    def test_no_promoted_model_returns_400(self, client):
        """If the bucket has no `current` pointer, return 400 (not 422)."""
        with patch("brain_api.routes.models_meta.get_bucket") as mock_get_bucket:
            fake_storage = MagicMock()
            fake_storage.read_current_version.return_value = None
            fake_bucket = MagicMock()
            fake_bucket.bucket_name = "sac_halal"
            fake_bucket.local_storage_class.return_value = fake_storage
            mock_get_bucket.return_value = fake_bucket

            response = client.get(
                "/models/active-symbols", params={"universe": "halal"}
            )

        assert response.status_code == 400
        assert "sac_halal" in response.json()["detail"]

"""Tests for /models/active-symbols endpoint.

Covers the mandatory ``universe`` query param (no default) introduced
to support the parallel A/B SAC workflows. Per AGENTS.md rule #1
("no silent fallbacks") the endpoint must 422 when callers omit
``universe`` so the two SAC buckets cannot accidentally share state.

Also pins the HF-aware read contract: the route now goes through
``load_current_artifacts_for_bucket`` so that under
``STORAGE_BACKEND=hf_first`` a freshly-deployed Pi recovers the symbol
slate from HF rather than 400-ing on an empty local cache. The legacy
400 cold-start contract is preserved via
``cold_start_status_code=400`` on the policy helper; transient HF
failures still surface as 503.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from brain_api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def _fake_artifacts(symbols: list[str], version: str) -> SimpleNamespace:
    """Build a minimal artifacts stub matching the SACArtifacts shape.

    The route only reads ``.symbol_order`` and ``.version``; we use
    ``SimpleNamespace`` so the stub stays narrow and any drift in the
    shape (e.g. accidentally reading ``.actor``) trips a clear
    ``AttributeError`` instead of a silent test pass.
    """
    return SimpleNamespace(
        symbol_order=symbols,
        version=version,
        metadata={"sac_schema_version": 3},
        v3_auxiliary=SimpleNamespace(training_cutoff_date="2026-04-30"),
    )


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

    def test_halal_filtered_routes_to_halal_filtered_bucket(self, client):
        """`?universe=halal_filtered` reads via the policy helper for that bucket."""
        with patch(
            "brain_api.routes.models_meta.load_current_artifacts_for_bucket"
        ) as mock_load:
            symbols = [f"S{i}" for i in range(15)]
            mock_load.return_value = _fake_artifacts(symbols, "v2026-05-01-aaa")

            response = client.get(
                "/models/active-symbols", params={"universe": "halal_filtered"}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["source_model"] == "sac_halal_filtered"
        assert data["model_version"] == "v2026-05-01-aaa"
        assert data["training_cutoff_date"] == "2026-04-30"
        assert data["sac_schema_version"] == 3
        assert len(data["symbols"]) == 15

        # The route MUST opt into the legacy 400 cold-start contract;
        # /inference/sac keeps the 503 default. Drift here would
        # silently revert this endpoint to 503 on cold start.
        kwargs = mock_load.call_args.kwargs
        assert kwargs["cold_start_status_code"] == 400
        assert kwargs["bucket"].bucket_name == "sac_halal_filtered"

    def test_halal_routes_to_halal_bucket(self, client):
        """`?universe=halal` reads via the policy helper for the halal bucket (variable size)."""
        with patch(
            "brain_api.routes.models_meta.load_current_artifacts_for_bucket"
        ) as mock_load:
            # halal bucket is variable size (10-15 typical); use 14 to
            # exercise the n-agnostic path explicitly.
            symbols = [f"H{i}" for i in range(14)]
            mock_load.return_value = _fake_artifacts(symbols, "v2026-05-01-bbb")

            response = client.get(
                "/models/active-symbols", params={"universe": "halal"}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["source_model"] == "sac_halal"
        assert data["model_version"] == "v2026-05-01-bbb"
        assert len(data["symbols"]) == 14

        kwargs = mock_load.call_args.kwargs
        assert kwargs["cold_start_status_code"] == 400
        assert kwargs["bucket"].bucket_name == "sac_halal"


class TestActiveSymbolsColdStartContract:
    """Pin the 400 vs 503 distinction at the route layer."""

    def test_no_promoted_model_anywhere_returns_400(self, client):
        """Genuine cold-start (no model in local OR HF) -> 400.

        Preserves the legacy ``"Train one first."`` contract even
        though the route now routes through the hf_first-aware policy
        layer. The policy raises HTTPException with the
        ``cold_start_status_code`` value the route passed in (400).
        """
        with patch(
            "brain_api.routes.models_meta.load_current_artifacts_for_bucket"
        ) as mock_load:
            mock_load.side_effect = HTTPException(
                status_code=400,
                detail=(
                    "hf_first: HF main is missing for SAC halal "
                    "(bucket 'sac_halal'). Cold-start: train locally first..."
                ),
            )

            response = client.get(
                "/models/active-symbols", params={"universe": "halal"}
            )

        assert response.status_code == 400
        assert "sac_halal" in response.json()["detail"]

    def test_transient_hf_failure_returns_503(self, client):
        """HF unreachable / download failed (NOT cold-start) -> 503.

        ``cold_start_status_code=400`` only downgrades the genuine
        "no model anywhere" cases. Transient or config 503s
        (HF down, hf_first without a repo, HF download failed) MUST
        propagate unchanged so an operator can distinguish "needs
        training" from "HF is having a bad day".
        """
        with patch(
            "brain_api.routes.models_meta.load_current_artifacts_for_bucket"
        ) as mock_load:
            mock_load.side_effect = HTTPException(
                status_code=503,
                detail=(
                    "No SAC halal model available: local empty and HF "
                    "(hajirazin/learnfinance-models-sac-it-heavy) failed: "
                    "ConnectionError"
                ),
            )

            response = client.get(
                "/models/active-symbols", params={"universe": "halal"}
            )

        assert response.status_code == 503
        # The 503 detail must surface the underlying broker so an
        # operator can tell this isn't a "train one first" condition.
        assert "HF" in response.json()["detail"]


class TestActiveSymbolsBucketResolutionStillFirst:
    """Bucket-resolution errors (422) take precedence over the policy call."""

    def test_unknown_universe_does_not_hit_policy_layer(self, client):
        """422 must short-circuit BEFORE we call the policy helper."""
        with patch(
            "brain_api.routes.models_meta.load_current_artifacts_for_bucket"
        ) as mock_load:
            response = client.get(
                "/models/active-symbols", params={"universe": "halal_new"}
            )

        assert response.status_code == 422
        # Policy helper must NOT be invoked for an unknown universe;
        # otherwise the policy could 503 on a typo and mask the 422.
        assert mock_load.call_count == 0

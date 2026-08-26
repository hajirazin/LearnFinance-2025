"""API tests for ppo_discovery routes (call the HTTP API)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from brain_api.core.ppo_discovery.synthetic import make_synthetic_state
from brain_api.main import app

client = TestClient(app)


def test_unknown_universe_is_422() -> None:
    response = client.post(
        "/train/ppo-discovery/preflight",
        json={"universe": "halal_filtered"},
    )
    assert response.status_code == 422
    response = client.post(
        "/inference/ppo-discovery",
        json={"state": {}, "state_digest": "sha256:x", "universe": "halal"},
    )
    assert response.status_code == 422
    response = client.get("/models/ppo-discovery/active", params={"universe": "halal"})
    assert response.status_code == 422


def test_digest_mismatch_is_422() -> None:
    state = make_synthetic_state()
    payload = state.to_dict()
    with patch(
        "brain_api.routes.inference.ppo_discovery.load_current_artifacts_for_bucket"
    ) as load:
        artifacts = MagicMock()
        artifacts.metadata = {
            "asset_feature_names": list(
                __import__(
                    "brain_api.core.ppo_discovery.config",
                    fromlist=["ASSET_FEATURE_NAMES"],
                ).ASSET_FEATURE_NAMES
            ),
            "global_feature_names": list(
                __import__(
                    "brain_api.core.ppo_discovery.config",
                    fromlist=["GLOBAL_FEATURE_NAMES"],
                ).GLOBAL_FEATURE_NAMES
            ),
            "news_required": True,
            "experiment_variant": "full",
        }
        load.return_value = artifacts
        response = client.post(
            "/inference/ppo-discovery",
            json={
                "state": payload,
                "state_digest": "sha256:deadbeef",
                "universe": "halal_new",
            },
        )
    assert response.status_code == 422
    assert "digest" in response.json()["detail"].lower()


def test_no_current_is_503() -> None:
    from fastapi import HTTPException

    state = make_synthetic_state()
    with patch(
        "brain_api.routes.inference.ppo_discovery.load_current_artifacts_for_bucket",
        side_effect=HTTPException(
            status_code=503, detail="no promoted ppo_discovery artifact"
        ),
    ):
        response = client.post(
            "/inference/ppo-discovery",
            json={
                "state": state.to_dict(),
                "state_digest": state.state_digest,
                "universe": "halal_new",
            },
        )
    assert response.status_code == 503


def test_promote_rejects_missing_approved_by() -> None:
    response = client.post(
        "/train/ppo-discovery/promote",
        json={"version": "v1", "expected_config_hash": "abc", "approved_by": ""},
    )
    assert response.status_code == 422


def test_promote_rejects_hash_mismatch() -> None:
    with patch(
        "brain_api.routes.training.ppo_discovery.promote.promote_ppo_discovery",
        side_effect=ValueError(
            "expected_config_hash does not match artifact config_hash"
        ),
    ):
        response = client.post(
            "/train/ppo-discovery/promote",
            json={
                "version": "v1",
                "expected_config_hash": "wrong",
                "approved_by": "razin",
            },
        )
    assert response.status_code == 422
    assert "hash" in response.json()["detail"].lower()


def test_promote_rejects_no_news_variant() -> None:
    with patch(
        "brain_api.routes.training.ppo_discovery.promote.promote_ppo_discovery",
        side_effect=ValueError("only experiment_variant='full' may be promoted"),
    ):
        response = client.post(
            "/train/ppo-discovery/promote",
            json={
                "version": "v-no-news",
                "expected_config_hash": "abc",
                "approved_by": "razin",
            },
        )
    assert response.status_code == 422
    assert "full" in response.json()["detail"].lower()


def test_incomplete_news_state_is_422() -> None:
    from brain_api.core.ppo_discovery.news_evidence import NewsEvidenceError

    with (
        patch(
            "brain_api.routes.signals.ppo_discovery.resolve_universe_snapshot"
        ) as snap,
        patch(
            "brain_api.routes.signals.ppo_discovery.materialize_news_evidence",
            side_effect=NewsEvidenceError("news query incomplete"),
        ),
    ):
        snap.return_value.sorted_symbols = ("AAPL", "MSFT")
        response = client.post(
            "/signals/ppo-discovery/state",
            json={
                "as_of": "2026-08-31T13:00:00+00:00",
                "run_id": "paper:halal_new:2026-08-31",
                "attempt": 1,
                "current_weights": {"CASH": 1.0},
                "universe": "halal_new",
            },
        )
    assert response.status_code == 422
    assert "incomplete" in response.json()["detail"].lower()


def test_etl_unknown_universe_422() -> None:
    response = client.post(
        "/etl/ppo-discovery/news-history",
        json={
            "start_date": "2026-01-01",
            "end_date": "2026-01-08",
            "universe": "halal",
        },
    )
    assert response.status_code == 422


@patch("brain_api.routes.training.ppo_discovery.preflight.resolve_universe_snapshot")
def test_preflight_halal_new_ok(mock_snap) -> None:
    mock_snap.return_value.universe = "halal_new"
    mock_snap.return_value.snapshot_sha256 = "sha256:abc"
    mock_snap.return_value.symbol_count = 12
    response = client.post(
        "/train/ppo-discovery/preflight",
        json={"universe": "halal_new", "experiment_id": "ci"},
    )
    assert response.status_code == 200
    assert response.json()["ready"] is True

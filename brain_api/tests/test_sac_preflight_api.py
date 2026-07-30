from fastapi.testclient import TestClient

from brain_api.core.sac.readiness import SACReadinessIssue, SACTrainingReadiness
from brain_api.main import app

client = TestClient(app)


def test_sac_preflight_returns_exact_missing_and_errors(monkeypatch):
    readiness = SACTrainingReadiness.from_issues(
        universe="halal_filtered",
        symbols=["AAA"],
        missing=[
            SACReadinessIssue(
                source="fundamentals",
                symbol="AAA",
                detail="filing availability unresolved",
                retryable=True,
            )
        ],
        errors=[
            SACReadinessIssue(
                source="news",
                symbol="AAA",
                detail="provider error",
                retryable=True,
            )
        ],
    )
    monkeypatch.setattr(
        "brain_api.routes.training.sac.preflight.assess_sac_training_readiness",
        lambda universe, *, force=False: readiness,
    )

    response = client.post("/train/sac/preflight", json={"universe": "halal_filtered"})

    assert response.status_code == 200
    assert response.json() == {
        "universe": "halal_filtered",
        "symbols": ["AAA"],
        "ready": False,
        "missing": [
            {
                "source": "fundamentals",
                "detail": "filing availability unresolved",
                "symbol": "AAA",
                "retryable": True,
            }
        ],
        "errors": [
            {
                "source": "news",
                "detail": "provider error",
                "symbol": "AAA",
                "retryable": True,
            }
        ],
    }


def test_sac_preflight_rejects_unknown_universe_at_api():
    response = client.post("/train/sac/preflight", json={"universe": "unknown"})

    assert response.status_code == 422

from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from brain_api.core.sac.readiness import SACReadinessIssue, SACTrainingReadiness
from brain_api.main import app
from brain_api.routes.training.sac import full as full_module
from brain_api.routes.training.sac import preflight as preflight_module

client = TestClient(app)


@pytest.fixture
def strict_preflight_dependencies(monkeypatch):
    """Provide complete inputs so each test can break one readiness contract."""
    symbols = ["AAA"]
    monkeypatch.setattr(
        preflight_module,
        "get_bucket",
        lambda model_type, universe: SimpleNamespace(
            symbols_resolver=lambda: symbols,
        ),
    )
    monkeypatch.setattr(
        preflight_module,
        "resolve_training_window",
        lambda: (date(2024, 1, 1), date(2024, 2, 1)),
    )
    monkeypatch.setattr(
        preflight_module,
        "SnapshotLocalStorage",
        lambda forecaster_type: object(),
    )
    monkeypatch.setattr(
        preflight_module,
        "ensure_snapshot_for_bucket",
        lambda **kwargs: True,
    )

    price_index = pd.date_range("2023-12-20", "2024-02-01", freq="D")
    prices = {
        "AAA": pd.DataFrame(
            {
                "open": 100.0,
                "close": 100.0,
            },
            index=price_index,
        )
    }
    monkeypatch.setattr(
        preflight_module,
        "load_prices_yfinance",
        lambda requested_symbols, start_date, end_date: prices,
    )

    monkeypatch.setattr(
        preflight_module,
        "require_weekly_news_coverage",
        lambda requested_symbols, weekly_cutoffs: None,
    )

    return prices


def test_sac_preflight_returns_exact_missing_and_errors(monkeypatch):
    readiness = SACTrainingReadiness.from_issues(
        universe="halal_filtered",
        symbols=["AAA"],
        missing=[
            SACReadinessIssue(
                source="prices",
                symbol="AAA",
                detail="price history incomplete",
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
                "source": "prices",
                "detail": "price history incomplete",
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
        "news_backfill_start": None,
        "news_backfill_end": None,
    }


def test_sac_preflight_rejects_unknown_universe_at_api():
    response = client.post("/train/sac/preflight", json={"universe": "unknown"})

    assert response.status_code == 422


def test_sac_preflight_reports_missing_price_history(
    monkeypatch,
    strict_preflight_dependencies,
):
    monkeypatch.setattr(
        preflight_module,
        "load_prices_yfinance",
        lambda requested_symbols, start_date, end_date: {},
    )

    response = client.post(
        "/train/sac/preflight",
        json={"universe": "halal_filtered", "force": True},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ready"] is False
    assert payload["missing"] == [
        {
            "source": "prices",
            "detail": "Missing daily price history for AAA",
            "symbol": "AAA",
            "retryable": True,
        }
    ]


def test_sac_preflight_old_news_schema_does_not_skip_readiness(
    monkeypatch,
    strict_preflight_dependencies,
):
    monkeypatch.setattr(
        preflight_module,
        "get_prior_metadata_for_bucket",
        lambda **kwargs: {
            "version": "v-old-news",
            "symbols": ["AAA"],
            "sac_schema_version": 3,
            "architecture": "masked_attention",
        },
    )
    monkeypatch.setattr(
        preflight_module,
        "load_prices_yfinance",
        lambda requested_symbols, start_date, end_date: {},
    )

    response = client.post("/train/sac/preflight", json={"universe": "halal_filtered"})

    assert response.status_code == 200
    assert response.json()["ready"] is False
    assert response.json()["missing"]


@pytest.mark.parametrize(
    ("issue_field", "expected_status"),
    [("missing", 409), ("errors", 503)],
)
def test_sac_full_rejects_unready_inputs_before_creating_job(
    monkeypatch,
    issue_field,
    expected_status,
):
    symbols = ["AAA"]
    bucket = SimpleNamespace(
        bucket_name="sac_halal_filtered",
        symbols_resolver=lambda: symbols,
        symbol_validator=None,
        local_storage_class=object,
    )
    issue = SACReadinessIssue(
        source="news",
        detail="provider observations incomplete",
        symbol="AAA",
        retryable=issue_field == "missing",
    )
    readiness = SACTrainingReadiness.from_issues(
        universe="halal_filtered",
        symbols=symbols,
        missing=[issue] if issue_field == "missing" else [],
        errors=[issue] if issue_field == "errors" else [],
    )
    monkeypatch.setattr(full_module, "get_bucket", lambda model_type, universe: bucket)
    monkeypatch.setattr(
        full_module, "get_prior_metadata_for_bucket", lambda **kwargs: None
    )
    monkeypatch.setattr(
        full_module,
        "resolve_training_window",
        lambda: (date(2024, 1, 1), date(2024, 2, 1)),
    )
    monkeypatch.setattr(full_module, "sac_compute_version", lambda *args: "v-test")
    monkeypatch.setattr(
        full_module, "try_load_existing_train_metadata", lambda **kwargs: None
    )
    monkeypatch.setattr(
        full_module,
        "assess_sac_training_readiness",
        lambda universe, *, force=False: readiness,
    )
    job_called = False

    def fail_if_job_created(*args, **kwargs):
        nonlocal job_called
        job_called = True
        raise AssertionError("training job must not be created")

    monkeypatch.setattr(full_module, "get_or_create_job", fail_if_job_created)

    response = client.post(
        "/train/sac/full",
        json={"universe": "halal_filtered", "force": True},
    )

    assert response.status_code == expected_status
    assert response.json()["detail"] == {
        "message": "SAC training inputs are not ready",
        "universe": "halal_filtered",
        "symbols": ["AAA"],
        "missing": [issue.to_dict()] if issue_field == "missing" else [],
        "errors": [issue.to_dict()] if issue_field == "errors" else [],
    }
    assert job_called is False

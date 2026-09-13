"""API tests for ppo_discovery routes (call the HTTP API)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from brain_api.core.ppo_discovery.synthetic import make_synthetic_state
from brain_api.main import app

client = TestClient(app)


def test_experience_label_uses_actual_nav_after_portfolio_drift() -> None:
    from brain_api.routes.experience_ppo_discovery import (
        PPOLabelRequest,
        label_ppo_discovery_experience,
    )

    record = MagicMock()
    record.run_id = "paper:halal_new:2026-01-05:ppo_discovery"
    record.week_start = "2026-01-05"
    record.week_end = "2026-01-12"
    record.model_type = "ppo_discovery"
    record.universe = "halal_new"
    record.state = {"current_weights": {"CASH": 1.0}}
    record.actual_weights = {"CASH": 1.0}
    record.nav_usd = 9_500.0
    storage = MagicMock()
    storage.load.return_value = record

    with patch(
        "brain_api.routes.experience_ppo_discovery.ppo_discovery_reward",
        return_value=(0.0, 0.0, 0.0, 0.0),
    ) as reward:
        result = label_ppo_discovery_experience(
            PPOLabelRequest(run_id="paper:halal_new:2026-01-05"),
            storage=storage,
        )

    assert result.records_labeled == 1
    assert result.errors == []
    assert reward.call_args.kwargs["nav_usd"] == 9_500.0
    assert reward.call_args.kwargs["config"].training_nav_usd == 10_000.0
    storage.update.assert_called_once_with(record)


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


def test_inference_rejects_state_payload_with_run_id() -> None:
    state = make_synthetic_state()
    payload = state.to_dict()
    payload["run_id"] = "paper:halal_new:2026-08-24"
    payload["attempt"] = 1
    response = client.post(
        "/inference/ppo-discovery",
        json={
            "state": payload,
            "state_digest": state.state_digest,
            "universe": "halal_new",
        },
    )
    assert response.status_code == 422
    assert "extra" in response.json()["detail"]


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
        json={"version": "v1", "approved_by": "", "expected_current_version": ""},
    )
    assert response.status_code == 422


def test_promote_does_not_pass_expected_config_hash() -> None:
    with patch(
        "brain_api.routes.training.ppo_discovery.promote.promote_ppo_discovery",
        return_value={
            "version": "v1",
            "approved_by": "razin",
            "promoted": True,
            "failure_reasons": [],
        },
    ) as promote:
        response = client.post(
            "/train/ppo-discovery/promote",
            json={
                "version": "v1",
                "approved_by": "razin",
                "expected_current_version": "",
            },
        )
    assert response.status_code == 200
    promote.assert_called_once()
    assert "expected_config_hash" not in promote.call_args.kwargs
    assert promote.call_args.kwargs["approved_by"] == "razin"
    assert promote.call_args.kwargs["expected_current_version"] == ""


def test_incomplete_news_state_is_422() -> None:
    from brain_api.news.errors import NewsCoverageMissing

    service = MagicMock()
    service.materialize.side_effect = NewsCoverageMissing("news query incomplete")
    with (
        patch(
            "brain_api.routes.signals.ppo_discovery.resolve_universe_snapshot"
        ) as snap,
        patch(
            "brain_api.routes.signals.ppo_discovery.require_monday_decision_cutoff",
            side_effect=lambda as_of: as_of,
        ),
        patch(
            "brain_api.routes.signals.ppo_discovery.get_news_service",
            return_value=service,
        ),
    ):
        snap.return_value.sorted_symbols = ("AAPL", "MSFT")
        response = client.post(
            "/signals/ppo-discovery/state",
            json={
                "as_of": "2026-08-24T09:00:00-04:00",
                "run_id": "paper:halal_new:2026-08-24",
                "attempt": 1,
                "current_weights": {"CASH": 1.0},
                "universe": "halal_new",
            },
        )
    assert response.status_code == 422
    assert "incomplete" in response.json()["detail"].lower()
    service.materialize.assert_called_once()


def test_legacy_ppo_news_history_route_is_gone() -> None:
    response = client.post(
        "/etl/news/backfill",
        json={
            "start": "2026-01-01T00:00:00+00:00",
            "end": "2026-01-08T00:00:00+00:00",
            "symbols": [],
        },
    )
    assert response.status_code in {404, 422}


@patch("brain_api.routes.training.ppo_discovery.preflight.resolve_universe_snapshot")
@patch("brain_api.routes.training.ppo_discovery.preflight.assess_price_readiness")
def test_preflight_halal_new_ok(mock_ready, mock_snap) -> None:
    mock_snap.return_value.universe = "halal_new"
    mock_snap.return_value.snapshot_sha256 = "sha256:abc"
    mock_snap.return_value.symbol_count = 12
    mock_snap.return_value.sorted_symbols = ("AAPL", "MSFT")
    mock_ready.return_value = {
        "ready": True,
        "issues": [],
        "exclusions": [],
        "session_hashes": {"AAPL": "a", "MSFT": "b"},
        "session_counts": {"AAPL": 300, "MSFT": 300},
        "eligible_symbol_count": 12,
        "vix_provenance": {
            "primary_provider": "yfinance",
            "fallback_provider": None,
            "fallback_dates": [],
            "source_url": None,
            "retrieved_at": None,
        },
        "index_end_date": "2026-08-14",
    }
    response = client.post(
        "/train/ppo-discovery/preflight",
        json={"universe": "halal_new", "experiment_id": "ci"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["ready"] is True
    assert payload["sorted_symbols"] == ["AAPL", "MSFT"]
    assert payload["snapshot_sha256"] == "sha256:abc"
    assert payload["exclusions"] == []
    mock_ready.assert_called_once()


@patch("brain_api.routes.training.ppo_discovery.full.resolve_universe_snapshot")
@patch("brain_api.routes.training.ppo_discovery.full.load_universe_snapshot")
def test_full_train_loads_persisted_snapshot(mock_load, mock_resolve) -> None:
    from brain_api.routes.training.ppo_discovery.full import (
        PPOTrainRequest,
        _load_training_snapshot,
    )

    mock_load.return_value = "loaded-snapshot"
    request = PPOTrainRequest(universe="halal_new", snapshot_sha256="sha256:frozen")
    assert _load_training_snapshot(request) == "loaded-snapshot"
    mock_load.assert_called_once_with("sha256:frozen")
    mock_resolve.assert_not_called()


def test_training_email_request_carries_evaluation() -> None:
    from jinja2 import Environment, FileSystemLoader

    from brain_api.routes.email.ppo_discovery import PPOTrainingEmailRequest
    from brain_api.routes.email.weekly_report import TEMPLATE_DIR

    request = PPOTrainingEmailRequest(
        version="v1",
        snapshot_sha256="sha256:abc",
        evaluation={
            "test_cagr": 0.21,
            "test_sharpe": 1.1,
            "selected_seed": 42,
        },
    )
    html = (
        Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), autoescape=True)
        .get_template("ppo_discovery_training_summary_email.html.j2")
        .render(**request.model_dump())
    )
    assert "0.21" in html
    assert "42" in html
    assert "full_ppo" not in html
    assert "IBKR Singapore Tiered costs at $10,000 training capital" in html
    assert "execution remains Alpaca" in html


def test_weekly_email_renders_portfolio_transition() -> None:
    from jinja2 import Environment, FileSystemLoader

    from brain_api.routes.email.ppo_discovery import PPOWeeklyEmailRequest
    from brain_api.routes.email.weekly_report import TEMPLATE_DIR

    request = PPOWeeklyEmailRequest(
        universe="halal_new",
        as_of="2026-09-07",
        model_version="v1",
        k=2,
        cash_weight=0.4,
        percentage_weights={"AAPL": 0.3, "MSFT": 0.3, "CASH": 0.4},
        explanations={
            "portfolio_transition": {
                "turnover": 0.25,
                "retained_count": 1,
                "entered_count": 1,
                "exited_count": 1,
                "replacement_fraction": 0.5,
                "stock_weight_cv": 0.01,
            }
        },
    )
    html = (
        Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), autoescape=True)
        .get_template("ppo_discovery_weekly_report_email.html.j2")
        .render(**request.model_dump())
    )
    assert "0.25" in html
    assert "Retained: 1" in html
    assert "Entered: 1" in html
    assert "Exited: 1" in html
    prompt = (
        Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), autoescape=False)
        .get_template("ppo_discovery_weekly_summary_prompt.j2")
        .render(
            universe="halal_new",
            model_version="v1",
            k=2,
            cash_weight=0.4,
            state_digest="abc",
            selected_symbols=["AAPL", "MSFT"],
            percentage_weights={"AAPL": 0.3, "MSFT": 0.3, "CASH": 0.4},
            explanations=request.explanations,
        )
    )
    assert "cannot alter or veto weights" in prompt
    assert "Replacement fraction" in prompt


def test_training_templates_render_test_metrics() -> None:
    from jinja2 import Environment, FileSystemLoader

    from brain_api.routes.email.weekly_report import TEMPLATE_DIR

    context = {
        "version": "v1",
        "promoted": False,
        "snapshot_sha256": "sha256:abc",
        "failure_reasons": [],
        "para_1_overall": "",
        "para_2_metrics": "",
        "para_3_recommendations": "",
        "evaluation": {
            "test_cagr": 0.21,
            "test_sharpe": 1.1,
            "test_max_drawdown": 0.08,
            "selected_seed": 42,
            "portfolio_diagnostics": {
                "summary": {
                    "mean_turnover": 0.10,
                    "mean_transaction_cost_bps": 12.0,
                    "mean_replacement_fraction": 0.2,
                }
            },
        },
    }
    html = (
        Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), autoescape=True)
        .get_template("ppo_discovery_training_summary_email.html.j2")
        .render(**context)
    )
    assert "1.1" in html
    assert "equal-weight-selected" not in html
    assert "no-cost-trained" not in html
    assert "full_ppo" not in html
    prompt = (
        Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), autoescape=False)
        .get_template("ppo_discovery_training_summary_prompt.j2")
        .render(**context)
    )
    assert "cannot alter or veto weights" in prompt
    assert "never auto-promotes" in prompt
    assert "test CAGR > 12%" in prompt


def test_backfill_job_exists_immediately_after_202(monkeypatch) -> None:
    jobs: dict[str, object] = {}

    class _Store:
        def get_job(self, job_id):
            return jobs.get(job_id)

        def upsert_job(self, job):
            jobs[job.job_id] = job

    store = _Store()
    monkeypatch.setattr("brain_api.routes.news_etl.get_news_store", lambda: store)
    monkeypatch.setattr(
        "brain_api.routes.news_etl._run_backfill", lambda *args, **kwargs: None
    )
    response = client.post(
        "/etl/news/backfill",
        json={
            "start": "2026-01-01T00:00:00+00:00",
            "end": "2026-01-20T00:00:00+00:00",
            "symbols": ["AAPL"],
        },
    )
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    assert response.json()["status"] == "pending"
    fetched = client.get(f"/etl/news/backfill/{job_id}")
    assert fetched.status_code == 200
    assert fetched.json()["job_id"] == job_id
    assert fetched.json()["status"] == "pending"

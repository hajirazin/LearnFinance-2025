"""Tests for POST /email/us-alpha-hrp-report."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from brain_api.main import app
from brain_api.routes.email.gmail import GmailConfigError

client = TestClient(app)


@pytest.fixture
def alpha_email_request():
    """Valid request payload for /email/us-alpha-hrp-report."""
    return {
        "summary": {
            "para_1_market_outlook": "Top forecasts cluster around US tech.",
            "para_2_selection_rationale": "Sticky kept 12 names, three new entrants.",
            "para_3_final_allocation": "HRP weights between 6.0% and 8.5%.",
            "para_4_risk_observations": "Watch K_hold across regime shifts.",
        },
        "stage1_top_scores": [
            {"symbol": f"S{i:03d}", "score": 5.0 - 0.1 * i, "rank": i + 1}
            for i in range(20)
        ],
        "model_version": "v2026-04-26-abc",
        "predicted_count": 380,
        "requested_count": 410,
        "selected_symbols": [f"S{i:03d}" for i in range(15)],
        "kept_count": 12,
        "fillers_count": 3,
        "evicted_from_previous": {"OLD1": "rank_out_of_hold"},
        "previous_year_week_used": "202617",
        "stage2": {
            "percentage_weights": {f"S{i:03d}": 100.0 / 15 for i in range(15)},
            "symbols_used": 15,
            "symbols_excluded": [],
            "lookback_days": 252,
            "as_of_date": "2026-04-28",
        },
        "universe": "halal_new",
        "top_n": 15,
        "hold_threshold": 30,
        "target_week_start": "2026-04-27",
        "target_week_end": "2026-05-01",
        "as_of_date": "2026-04-28",
        "order_results": {
            "orders_submitted": 14,
            "orders_failed": 1,
            "skipped": False,
        },
        "skipped": False,
    }


class TestUSAlphaHRPReportEmailEndpoint:
    @patch("brain_api.routes.email.weekly_report.send_html_email")
    def test_happy_path_with_orders(self, mock_send_email, alpha_email_request):
        mock_send_email.return_value = True
        response = client.post(
            "/email/us-alpha-hrp-report",
            json=alpha_email_request,
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["is_success"] is True
        assert "US Alpha-HRP Portfolio Analysis" in data["subject"]
        assert "2026-04-27" in data["subject"]

        body = data["body"]
        assert "AI Analysis Summary" in body
        assert "Stage 1: Alpha Screen" in body
        assert "Stage 2: HRP Allocation" in body
        assert "Alpaca Order Execution" in body
        assert "Rank-band Sticky Selection" in body
        assert "halal_new" in body
        assert "14" in body
        assert "Run Skipped" not in body

    @patch("brain_api.routes.email.weekly_report.send_html_email")
    def test_skipped_path_hides_orders(self, mock_send_email, alpha_email_request):
        mock_send_email.return_value = True
        alpha_email_request["skipped"] = True
        response = client.post(
            "/email/us-alpha-hrp-report",
            json=alpha_email_request,
        )
        assert response.status_code == 200
        data = response.json()
        assert "US Alpha-HRP Skipped" in data["subject"]
        body = data["body"]
        assert "Run Skipped" in body
        assert "AI Analysis Summary" not in body
        assert "Alpaca Order Execution" not in body

    @patch("brain_api.routes.email.weekly_report.send_html_email")
    def test_no_order_results_renders(self, mock_send_email, alpha_email_request):
        mock_send_email.return_value = True
        alpha_email_request["order_results"] = None
        response = client.post(
            "/email/us-alpha-hrp-report",
            json=alpha_email_request,
        )
        assert response.status_code == 200
        body = response.json()["body"]
        assert "Stage 2: HRP Allocation" in body
        assert "Alpaca Order Execution" not in body

    @patch("brain_api.routes.email.weekly_report.send_html_email")
    def test_smtp_failure_returns_503(self, mock_send_email, alpha_email_request):
        mock_send_email.side_effect = Exception("SMTP down")
        response = client.post(
            "/email/us-alpha-hrp-report",
            json=alpha_email_request,
        )
        assert response.status_code == 503
        assert "Failed to send email" in response.json()["detail"]

    @patch("brain_api.routes.email.weekly_report.send_html_email")
    def test_gmail_config_error_returns_500(self, mock_send_email, alpha_email_request):
        mock_send_email.side_effect = GmailConfigError("GMAIL_USER is required")
        response = client.post(
            "/email/us-alpha-hrp-report",
            json=alpha_email_request,
        )
        assert response.status_code == 500

    def test_missing_required_field_returns_422(self):
        response = client.post(
            "/email/us-alpha-hrp-report",
            json={
                "summary": {"para_1_market_outlook": "x"},
                "universe": "halal_new",
                "top_n": 15,
                "hold_threshold": 30,
                "target_week_start": "2026-04-27",
                "target_week_end": "2026-05-01",
                "as_of_date": "2026-04-28",
            },
        )
        assert response.status_code == 422

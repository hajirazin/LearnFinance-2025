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
            "para_5_stage_transition_insight": "S012 jumped from alpha rank 12 to HRP weight rank 1 due to low correlation with basket peers.",
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

    @patch("brain_api.routes.email.weekly_report.send_html_email")
    def test_with_per_order_detail_table(self, mock_send_email, alpha_email_request):
        """Detailed order table renders when ``orders`` list is populated.

        The plumbing test: brain_api must surface symbol/qty/price/stop
        for every row the workflow shipped, and never substitute a flat
        percent when ATR is missing (AGENTS.md rule #1).
        """
        mock_send_email.return_value = True
        alpha_email_request["order_results"] = {
            "orders_submitted": 2,
            "orders_failed": 0,
            "skipped": False,
            "orders": [
                {
                    "symbol": "S001",
                    "side": "buy",
                    "qty": 10.5,
                    "current_price": 100.0,
                    "trade_value": 1050.0,
                    "stop_loss_price": 94.0,
                    "stop_loss_distance_pct": 0.06,
                    "stop_loss_reason": "atr14",
                    "client_order_id": "paper:2026-04-27:attempt-1:S001:buy",
                    "submission_status": "submitted",
                },
                {
                    "symbol": "S002",
                    "side": "sell",
                    "qty": 3.0,
                    "current_price": 50.0,
                    "trade_value": 150.0,
                    "stop_loss_price": None,
                    "stop_loss_distance_pct": None,
                    "stop_loss_reason": "sell_no_stop",
                    "client_order_id": "paper:2026-04-27:attempt-1:S002:sell",
                    "submission_status": "submitted",
                },
            ],
        }
        response = client.post(
            "/email/us-alpha-hrp-report",
            json=alpha_email_request,
        )
        assert response.status_code == 200
        body = response.json()["body"]
        assert "Order Execution Detail" in body
        assert "S001" in body
        assert "S002" in body
        # Stop-loss price rendered for the buy.
        assert "$94.00" in body
        # Sell row: em-dash, NOT a flat percent fallback.
        assert "sell_no_stop" not in body  # reason string itself isn't shown
        # Buys with ATR show the distance %.
        assert "6.0%" in body or "-6.0%" in body

    @patch("brain_api.routes.email.weekly_report.send_html_email")
    def test_atr_unavailable_renders_n_a_not_flat_percent(
        self, mock_send_email, alpha_email_request
    ):
        """Missing ATR shows ``n/a (no ATR)`` instead of a faked stop.

        AGENTS.md rule #1: never silently substitute a flat percent.
        """
        mock_send_email.return_value = True
        alpha_email_request["order_results"] = {
            "orders_submitted": 1,
            "orders_failed": 0,
            "skipped": False,
            "orders": [
                {
                    "symbol": "S099",
                    "side": "buy",
                    "qty": 1.0,
                    "current_price": 100.0,
                    "trade_value": 100.0,
                    "stop_loss_price": None,
                    "stop_loss_distance_pct": None,
                    "stop_loss_reason": "atr_unavailable",
                    "client_order_id": "paper:2026-04-27:attempt-1:S099:buy",
                    "submission_status": "submitted",
                },
            ],
        }
        response = client.post(
            "/email/us-alpha-hrp-report",
            json=alpha_email_request,
        )
        assert response.status_code == 200
        body = response.json()["body"]
        assert "n/a (no ATR)" in body

    @patch("brain_api.routes.email.weekly_report.send_html_email")
    def test_with_prior_allocation_block(self, mock_send_email, alpha_email_request):
        """ "Going Into This Week" block renders with the source label."""
        mock_send_email.return_value = True
        alpha_email_request["prior_allocation"] = {
            "weights": {"S001": 0.10, "S002": 0.05, "CASH": 0.85},
            "source_label": "live Alpaca account: hrp",
            "as_of": "2026-04-21",
        }
        response = client.post(
            "/email/us-alpha-hrp-report",
            json=alpha_email_request,
        )
        assert response.status_code == 200
        body = response.json()["body"]
        assert "Going Into This Week" in body
        assert "live Alpaca account: hrp" in body
        # Symbols from the prior snapshot show up in the delta table.
        assert "S001" in body
        assert "CASH" in body

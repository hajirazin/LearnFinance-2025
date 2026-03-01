"""Tests for India weekly email flow.

This module tests:
- Full flow execution with mocked tasks
- Sequential dependency (failure propagation)
- India-specific task behavior (generate_india_summary, send_india_weekly_email)
"""

from unittest.mock import MagicMock, patch

import pytest

from flows.india_weekly_email import india_weekly_email_flow
from flows.models import (
    HRPAllocationResponse,
    WeeklyReportEmailResponse,
    WeeklySummaryResponse,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_universe_data():
    """Mock response from GET /universe/halal_india."""
    return {
        "stocks": [
            {"symbol": "RELIANCE", "name": "Reliance Industries", "factor_score": 0.85},
            {
                "symbol": "TCS",
                "name": "Tata Consultancy Services",
                "factor_score": 0.82,
            },
            {"symbol": "INFY", "name": "Infosys Ltd", "factor_score": 0.78},
        ],
        "source": "nifty_500_shariah",
        "symbol_suffix": ".NS",
        "total_stocks": 120,
        "total_scored": 95,
        "top_n": 15,
    }


@pytest.fixture
def mock_hrp_allocation():
    """Mock HRP allocation response for India."""
    return HRPAllocationResponse(
        universe="halal_india",
        percentage_weights={
            "RELIANCE.NS": 25.0,
            "TCS.NS": 20.0,
            "INFY.NS": 15.0,
        },
        symbols_used=3,
        symbols_excluded=[],
        as_of_date="2026-03-02",
    )


@pytest.fixture
def mock_india_summary():
    """Mock India LLM summary response."""
    return WeeklySummaryResponse(
        summary={
            "para_1_portfolio_overview": "HRP distributed weights across 3 stocks.",
            "para_2_concentration_analysis": "Top 3 stocks hold 60% of the portfolio.",
            "para_3_risk_observations": "IT sector is overweight in this allocation.",
        },
        provider="openai",
        model_used="gpt-4o-mini",
        tokens_used=350,
    )


@pytest.fixture
def mock_india_email():
    """Mock India email response."""
    return WeeklyReportEmailResponse(
        is_success=True,
        subject="India Weekly Portfolio Analysis (2026-03-02 -> 2026-03-06)",
        body="<html><body>India report</body></html>",
    )


# =============================================================================
# Full Flow Tests
# =============================================================================


class TestIndiaFullFlow:
    """Test full India flow execution with mocked tasks."""

    @patch("flows.india_weekly_email.send_india_weekly_email")
    @patch("flows.india_weekly_email.generate_india_summary")
    @patch("flows.india_weekly_email.allocate_hrp")
    @patch("flows.india_weekly_email.get_halal_india_universe")
    def test_full_flow_success(
        self,
        mock_get_universe,
        mock_allocate_hrp,
        mock_gen_summary,
        mock_send_email,
        mock_universe_data,
        mock_hrp_allocation,
        mock_india_summary,
        mock_india_email,
    ):
        """Test successful end-to-end India flow execution."""
        mock_get_universe.return_value = mock_universe_data
        mock_allocate_hrp.return_value = mock_hrp_allocation
        mock_gen_summary.return_value = mock_india_summary
        mock_send_email.return_value = mock_india_email

        result = india_weekly_email_flow()

        assert result["universe_stocks"] == 3
        assert result["hrp_symbols"] == 3
        assert result["summary_provider"] == "openai"
        assert result["email"]["is_success"] is True
        assert "India" in result["email"]["subject"]

        mock_get_universe.assert_called_once()
        mock_allocate_hrp.assert_called_once()
        mock_gen_summary.assert_called_once_with(mock_hrp_allocation)
        mock_send_email.assert_called_once()

    @patch("flows.india_weekly_email.send_india_weekly_email")
    @patch("flows.india_weekly_email.generate_india_summary")
    @patch("flows.india_weekly_email.allocate_hrp")
    @patch("flows.india_weekly_email.get_halal_india_universe")
    def test_hrp_called_with_halal_india_universe(
        self,
        mock_get_universe,
        mock_allocate_hrp,
        mock_gen_summary,
        mock_send_email,
        mock_universe_data,
        mock_hrp_allocation,
        mock_india_summary,
        mock_india_email,
    ):
        """Test that HRP is called with universe='halal_india'."""
        mock_get_universe.return_value = mock_universe_data
        mock_allocate_hrp.return_value = mock_hrp_allocation
        mock_gen_summary.return_value = mock_india_summary
        mock_send_email.return_value = mock_india_email

        india_weekly_email_flow()

        call_kwargs = mock_allocate_hrp.call_args
        assert call_kwargs.kwargs["universe"] == "halal_india"

    @patch("flows.india_weekly_email.send_india_weekly_email")
    @patch("flows.india_weekly_email.generate_india_summary")
    @patch("flows.india_weekly_email.allocate_hrp")
    @patch("flows.india_weekly_email.get_halal_india_universe")
    def test_email_receives_correct_dates(
        self,
        mock_get_universe,
        mock_allocate_hrp,
        mock_gen_summary,
        mock_send_email,
        mock_universe_data,
        mock_hrp_allocation,
        mock_india_summary,
        mock_india_email,
    ):
        """Test that email task receives correct date params."""
        mock_get_universe.return_value = mock_universe_data
        mock_allocate_hrp.return_value = mock_hrp_allocation
        mock_gen_summary.return_value = mock_india_summary
        mock_send_email.return_value = mock_india_email

        result = india_weekly_email_flow()

        call_kwargs = mock_send_email.call_args.kwargs
        assert call_kwargs["target_week_start"] == result["target_week_start"]
        assert call_kwargs["target_week_end"] == result["target_week_end"]
        assert call_kwargs["as_of_date"] == result["as_of_date"]

    @patch("flows.india_weekly_email.send_india_weekly_email")
    @patch("flows.india_weekly_email.generate_india_summary")
    @patch("flows.india_weekly_email.allocate_hrp")
    @patch("flows.india_weekly_email.get_halal_india_universe")
    def test_target_week_end_is_friday(
        self,
        mock_get_universe,
        mock_allocate_hrp,
        mock_gen_summary,
        mock_send_email,
        mock_universe_data,
        mock_hrp_allocation,
        mock_india_summary,
        mock_india_email,
    ):
        """Test that target_week_end is 4 days after target_week_start."""
        mock_get_universe.return_value = mock_universe_data
        mock_allocate_hrp.return_value = mock_hrp_allocation
        mock_gen_summary.return_value = mock_india_summary
        mock_send_email.return_value = mock_india_email

        result = india_weekly_email_flow()

        from datetime import datetime

        start = datetime.strptime(result["target_week_start"], "%Y-%m-%d")
        end = datetime.strptime(result["target_week_end"], "%Y-%m-%d")
        assert (end - start).days == 4


# =============================================================================
# Failure Propagation Tests
# =============================================================================


class TestFailurePropagation:
    """Test that failures propagate correctly through the sequential pipeline."""

    @patch("flows.india_weekly_email.send_india_weekly_email")
    @patch("flows.india_weekly_email.generate_india_summary")
    @patch("flows.india_weekly_email.allocate_hrp")
    @patch("flows.india_weekly_email.get_halal_india_universe")
    def test_universe_failure_stops_flow(
        self,
        mock_get_universe,
        mock_allocate_hrp,
        mock_gen_summary,
        mock_send_email,
    ):
        """If universe fetch fails, HRP should not be called."""
        mock_get_universe.side_effect = Exception("NSE API down")

        with pytest.raises(Exception, match="NSE API down"):
            india_weekly_email_flow()

        mock_allocate_hrp.assert_not_called()
        mock_gen_summary.assert_not_called()
        mock_send_email.assert_not_called()

    @patch("flows.india_weekly_email.send_india_weekly_email")
    @patch("flows.india_weekly_email.generate_india_summary")
    @patch("flows.india_weekly_email.allocate_hrp")
    @patch("flows.india_weekly_email.get_halal_india_universe")
    def test_hrp_failure_stops_summary_and_email(
        self,
        mock_get_universe,
        mock_allocate_hrp,
        mock_gen_summary,
        mock_send_email,
        mock_universe_data,
    ):
        """If HRP fails, summary and email should not run."""
        mock_get_universe.return_value = mock_universe_data
        mock_allocate_hrp.side_effect = Exception("HRP allocation failed")

        with pytest.raises(Exception, match="HRP allocation failed"):
            india_weekly_email_flow()

        mock_get_universe.assert_called_once()
        mock_gen_summary.assert_not_called()
        mock_send_email.assert_not_called()

    @patch("flows.india_weekly_email.send_india_weekly_email")
    @patch("flows.india_weekly_email.generate_india_summary")
    @patch("flows.india_weekly_email.allocate_hrp")
    @patch("flows.india_weekly_email.get_halal_india_universe")
    def test_summary_failure_stops_email(
        self,
        mock_get_universe,
        mock_allocate_hrp,
        mock_gen_summary,
        mock_send_email,
        mock_universe_data,
        mock_hrp_allocation,
    ):
        """If LLM summary fails, email should not be sent."""
        mock_get_universe.return_value = mock_universe_data
        mock_allocate_hrp.return_value = mock_hrp_allocation
        mock_gen_summary.side_effect = Exception("OpenAI rate limited")

        with pytest.raises(Exception, match="OpenAI rate limited"):
            india_weekly_email_flow()

        mock_get_universe.assert_called_once()
        mock_allocate_hrp.assert_called_once()
        mock_send_email.assert_not_called()


# =============================================================================
# Task-Level Tests
# =============================================================================


class TestIndiaTaskBehavior:
    """Test individual India tasks call correct endpoints with correct payloads."""

    @patch("flows.tasks.reporting.get_run_logger")
    @patch("flows.tasks.reporting.get_client")
    def test_generate_india_summary_calls_correct_endpoint(
        self, mock_get_client, mock_get_logger
    ):
        """Test generate_india_summary posts to /llm/india-weekly-summary."""
        from flows.tasks.reporting import generate_india_summary

        mock_get_logger.return_value = MagicMock()
        mock_client = MagicMock()
        mock_get_client.return_value.__enter__.return_value = mock_client
        mock_client.post.return_value.json.return_value = {
            "summary": {"para_1_portfolio_overview": "Test summary."},
            "provider": "openai",
            "model_used": "gpt-4o-mini",
            "tokens_used": 100,
        }

        hrp = HRPAllocationResponse(
            universe="halal_india",
            percentage_weights={"RELIANCE.NS": 30.0, "TCS.NS": 20.0},
            symbols_used=2,
            symbols_excluded=[],
            as_of_date="2026-03-02",
        )

        generate_india_summary.fn(hrp)

        call_args = mock_client.post.call_args
        assert call_args.args[0] == "/llm/india-weekly-summary"
        json_body = call_args.kwargs["json"]
        assert "hrp" in json_body
        assert json_body["hrp"]["universe"] == "halal_india"

    @patch("flows.tasks.reporting.get_run_logger")
    @patch("flows.tasks.reporting.get_client")
    def test_send_india_weekly_email_calls_correct_endpoint(
        self, mock_get_client, mock_get_logger
    ):
        """Test send_india_weekly_email posts to /email/india-weekly-report."""
        from flows.tasks.reporting import send_india_weekly_email

        mock_get_logger.return_value = MagicMock()
        mock_client = MagicMock()
        mock_get_client.return_value.__enter__.return_value = mock_client
        mock_client.post.return_value.json.return_value = {
            "is_success": True,
            "subject": "India Weekly Portfolio Analysis (2026-03-02 -> 2026-03-06)",
            "body": "<html>report</html>",
        }

        summary = WeeklySummaryResponse(
            summary={"para_1_portfolio_overview": "Test."},
            provider="openai",
            model_used="gpt-4o-mini",
            tokens_used=100,
        )
        hrp = HRPAllocationResponse(
            universe="halal_india",
            percentage_weights={"RELIANCE.NS": 30.0},
            symbols_used=1,
            symbols_excluded=[],
            as_of_date="2026-03-02",
        )

        send_india_weekly_email.fn(
            summary=summary,
            hrp=hrp,
            target_week_start="2026-03-02",
            target_week_end="2026-03-06",
            as_of_date="2026-03-02",
        )

        call_args = mock_client.post.call_args
        assert call_args.args[0] == "/email/india-weekly-report"
        json_body = call_args.kwargs["json"]
        assert json_body["target_week_start"] == "2026-03-02"
        assert json_body["target_week_end"] == "2026-03-06"
        assert json_body["as_of_date"] == "2026-03-02"
        assert "summary" in json_body
        assert "hrp" in json_body
        assert json_body["hrp"]["universe"] == "halal_india"

    @patch("flows.tasks.inference.get_run_logger")
    @patch("flows.tasks.inference.get_client")
    def test_get_halal_india_universe_calls_correct_endpoint(
        self, mock_get_client, mock_get_logger
    ):
        """Test get_halal_india_universe calls GET /universe/halal_india."""
        from flows.tasks.inference import get_halal_india_universe

        mock_get_logger.return_value = MagicMock()
        mock_client = MagicMock()
        mock_get_client.return_value.__enter__.return_value = mock_client
        mock_client.get.return_value.json.return_value = {
            "stocks": [{"symbol": "RELIANCE"}],
            "source": "nifty_500_shariah",
        }

        result = get_halal_india_universe.fn()

        mock_client.get.assert_called_once_with("/universe/halal_india")
        assert len(result["stocks"]) == 1

    @patch("flows.tasks.inference.get_run_logger")
    @patch("flows.tasks.inference.get_client")
    def test_allocate_hrp_passes_universe_param(self, mock_get_client, mock_get_logger):
        """Test allocate_hrp passes the universe parameter in the JSON body."""
        from flows.tasks.inference import allocate_hrp

        mock_get_logger.return_value = MagicMock()
        mock_client = MagicMock()
        mock_get_client.return_value.__enter__.return_value = mock_client
        mock_client.post.return_value.json.return_value = {
            "universe": "halal_india",
            "percentage_weights": {"RELIANCE.NS": 30.0},
            "symbols_used": 1,
            "symbols_excluded": [],
            "as_of_date": "2026-03-02",
        }

        allocate_hrp.fn(as_of_date="2026-03-02", universe="halal_india")

        call_args = mock_client.post.call_args
        json_body = call_args.kwargs["json"]
        assert json_body["universe"] == "halal_india"

    @patch("flows.tasks.inference.get_run_logger")
    @patch("flows.tasks.inference.get_client")
    def test_allocate_hrp_defaults_to_halal_filtered(
        self, mock_get_client, mock_get_logger
    ):
        """Test allocate_hrp defaults to halal_filtered for backward compat."""
        from flows.tasks.inference import allocate_hrp

        mock_get_logger.return_value = MagicMock()
        mock_client = MagicMock()
        mock_get_client.return_value.__enter__.return_value = mock_client
        mock_client.post.return_value.json.return_value = {
            "universe": "halal_filtered",
            "percentage_weights": {"AAPL": 20.0},
            "symbols_used": 1,
            "symbols_excluded": [],
            "as_of_date": "2026-03-02",
        }

        allocate_hrp.fn(as_of_date="2026-03-02")

        call_args = mock_client.post.call_args
        json_body = call_args.kwargs["json"]
        assert json_body["universe"] == "halal_filtered"

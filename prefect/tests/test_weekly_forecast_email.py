"""Tests for weekly forecast email flow.

This module tests:
- Pydantic models for forecast email responses
- Full flow execution with mocked tasks
- Skip logic when algorithms have open orders
- HRP percentage to decimal weight conversion
- Parallel execution phases
"""

from unittest.mock import MagicMock, patch

import pytest

from flows.models import (
    AlpacaPortfolioResponse,
    FundamentalsResponse,
    GenerateOrdersResponse,
    HalalUniverseResponse,
    HRPAllocationResponse,
    LSTMInferenceResponse,
    NewsSignalResponse,
    OrderModel,
    OrderSubmitResult,
    PatchTSTInferenceResponse,
    PositionModel,
    PPOInferenceResponse,
    SACInferenceResponse,
    SkippedAllocation,
    SkippedOrdersResponse,
    SkippedSubmitResponse,
    SubmitOrdersResponse,
    WeeklyReportEmailResponse,
    WeeklySummaryResponse,
)
from flows.weekly_forecast_email import weekly_forecast_email_flow

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_universe_response():
    """Mock response for halal universe endpoint."""
    return HalalUniverseResponse(
        stocks=[
            {
                "symbol": f"SYM{i}",
                "name": f"Stock {i}",
                "max_weight": 5.0,
                "sources": ["SPUS"],
            }
            for i in range(25)  # More than 20 to test slicing
        ],
        etfs_used=["SPUS", "HLAL"],
        total_stocks=25,
        fetched_at="2026-02-05T12:00:00+00:00",
    )


@pytest.fixture
def mock_portfolio_no_open_orders():
    """Mock portfolio with no open orders."""
    return AlpacaPortfolioResponse(
        cash=10000.0,
        positions=[
            PositionModel(symbol="AAPL", qty=10.0, market_value=1750.0),
            PositionModel(symbol="MSFT", qty=5.0, market_value=2000.0),
        ],
        open_orders_count=0,
    )


@pytest.fixture
def mock_portfolio_with_open_orders():
    """Mock portfolio with open orders."""
    return AlpacaPortfolioResponse(
        cash=10000.0,
        positions=[
            PositionModel(symbol="AAPL", qty=10.0, market_value=1750.0),
        ],
        open_orders_count=3,
    )


@pytest.fixture
def mock_lstm_response():
    """Mock LSTM inference response."""
    return LSTMInferenceResponse(
        predictions=[
            {
                "symbol": "AAPL",
                "predicted_weekly_return_pct": 2.5,
                "direction": "up",
                "has_enough_history": True,
            },
            {
                "symbol": "MSFT",
                "predicted_weekly_return_pct": -1.0,
                "direction": "down",
                "has_enough_history": True,
            },
        ],
        model_version="v1.0.0",
        as_of_date="2026-02-05",
        target_week_start="2026-02-10",
        target_week_end="2026-02-14",
    )


@pytest.fixture
def mock_patchtst_response():
    """Mock PatchTST inference response."""
    return PatchTSTInferenceResponse(
        predictions=[
            {
                "symbol": "AAPL",
                "predicted_weekly_return_pct": 3.0,
                "direction": "up",
                "has_enough_history": True,
            },
            {
                "symbol": "MSFT",
                "predicted_weekly_return_pct": -0.5,
                "direction": "down",
                "has_enough_history": True,
            },
        ],
        model_version="v1.0.0",
        as_of_date="2026-02-05",
        signals_used=["price", "sentiment", "fundamentals"],
        target_week_start="2026-02-10",
        target_week_end="2026-02-14",
    )


@pytest.fixture
def mock_ppo_allocation():
    """Mock PPO inference response."""
    return PPOInferenceResponse(
        target_weights={"AAPL": 0.30, "MSFT": 0.20, "CASH": 0.50},
        turnover=0.15,
        model_version="v1.0.0",
        target_week_start="2026-02-10",
        target_week_end="2026-02-14",
    )


@pytest.fixture
def mock_sac_allocation():
    """Mock SAC inference response."""
    return SACInferenceResponse(
        target_weights={"AAPL": 0.25, "MSFT": 0.25, "CASH": 0.50},
        turnover=0.12,
        model_version="v1.0.0",
        target_week_start="2026-02-10",
        target_week_end="2026-02-14",
    )


@pytest.fixture
def mock_hrp_allocation():
    """Mock HRP allocation response with percentage weights."""
    return HRPAllocationResponse(
        percentage_weights={"AAPL": 20.0, "MSFT": 15.0, "GOOGL": 10.0},  # percentages!
        symbols_used=3,
        symbols_excluded=[],
        as_of_date="2026-02-05",
    )


@pytest.fixture
def mock_news_response():
    """Mock news sentiment response."""
    return NewsSignalResponse(
        per_symbol=[
            {"symbol": "AAPL", "sentiment_score": 0.5, "article_count": 10},
            {"symbol": "MSFT", "sentiment_score": -0.2, "article_count": 5},
        ],
        as_of_date="2026-02-05",
    )


@pytest.fixture
def mock_fundamentals_response():
    """Mock fundamentals response."""
    return FundamentalsResponse(
        per_symbol=[
            {
                "symbol": "AAPL",
                "ratios": {"gross_margin": 0.42, "operating_margin": 0.30},
            },
            {
                "symbol": "MSFT",
                "ratios": {"gross_margin": 0.68, "operating_margin": 0.42},
            },
        ],
        as_of_date="2026-02-05",
    )


@pytest.fixture
def mock_generate_orders_response():
    """Mock order generation response."""
    return GenerateOrdersResponse(
        orders=[
            OrderModel(
                client_order_id="paper:2026-02-05:attempt-1:AAPL:BUY",
                symbol="AAPL",
                side="buy",
                qty=5.0,
                type="limit",
                limit_price=175.50,
                time_in_force="day",
            ),
        ],
        summary={
            "buys": 1,
            "sells": 0,
            "total_buy_value": 877.50,
            "total_sell_value": 0,
            "turnover_pct": 8.8,
            "skipped_small_orders": 0,
        },
        prices_used={"AAPL": 175.50},
    )


@pytest.fixture
def mock_submit_orders_response():
    """Mock order submission response."""
    return SubmitOrdersResponse(
        account="ppo",
        orders_submitted=1,
        orders_failed=0,
        skipped=False,
        results=[
            OrderSubmitResult(
                id="order-123",
                client_order_id="paper:2026-02-05:attempt-1:AAPL:BUY",
                symbol="AAPL",
                status="accepted",
            )
        ],
    )


@pytest.fixture
def mock_summary_response():
    """Mock weekly summary response."""
    return WeeklySummaryResponse(
        summary={
            "overview": "Weekly portfolio analysis complete.",
            "allocations": "PPO and SAC generated allocations.",
            "outlook": "Positive outlook for next week.",
        },
        provider="openai",
        model_used="gpt-4o-mini",
        tokens_used=500,
    )


@pytest.fixture
def mock_email_response():
    """Mock email response."""
    return WeeklyReportEmailResponse(
        is_success=True,
        subject="Weekly Forecast Report: 2026-02-05",
        body="<html><body>Report content</body></html>",
    )


# =============================================================================
# Model Tests
# =============================================================================


class TestForecastEmailModels:
    """Tests for Pydantic models used in forecast email flow."""

    def test_alpaca_portfolio_response(self, mock_portfolio_no_open_orders):
        """Test AlpacaPortfolioResponse model."""
        assert mock_portfolio_no_open_orders.cash == 10000.0
        assert len(mock_portfolio_no_open_orders.positions) == 2
        assert mock_portfolio_no_open_orders.open_orders_count == 0

    def test_portfolio_with_open_orders(self, mock_portfolio_with_open_orders):
        """Test portfolio with open orders."""
        assert mock_portfolio_with_open_orders.open_orders_count == 3

    def test_hrp_allocation_percentage_weights(self, mock_hrp_allocation):
        """Test HRP allocation returns percentage weights (not decimals)."""
        assert mock_hrp_allocation.percentage_weights["AAPL"] == 20.0  # 20%, not 0.20
        assert mock_hrp_allocation.percentage_weights["MSFT"] == 15.0  # 15%, not 0.15

    def test_ppo_allocation_decimal_weights(self, mock_ppo_allocation):
        """Test PPO allocation returns decimal weights."""
        assert mock_ppo_allocation.target_weights["AAPL"] == 0.30  # 30% as decimal
        assert mock_ppo_allocation.target_weights["MSFT"] == 0.20  # 20% as decimal

    def test_skipped_allocation(self):
        """Test SkippedAllocation placeholder model."""
        skipped = SkippedAllocation(algorithm="ppo")
        assert skipped.skipped is True
        assert skipped.algorithm == "ppo"
        assert skipped.target_weights == {}
        assert skipped.reason == "Open orders exist"

    def test_skipped_orders_response(self):
        """Test SkippedOrdersResponse placeholder model."""
        skipped = SkippedOrdersResponse(skipped=True, algorithm="sac")
        assert skipped.skipped is True
        assert skipped.orders == []

    def test_skipped_submit_response(self):
        """Test SkippedSubmitResponse placeholder model."""
        skipped = SkippedSubmitResponse(account="hrp")
        assert skipped.skipped is True
        assert skipped.orders_submitted == 0

    def test_lstm_response_target_dates(self, mock_lstm_response):
        """Test LSTM response includes target week dates."""
        assert mock_lstm_response.target_week_start == "2026-02-10"
        assert mock_lstm_response.target_week_end == "2026-02-14"


# =============================================================================
# Full Flow Tests
# =============================================================================


class TestFullFlow:
    """Test full flow execution with mocked tasks."""

    @patch("flows.weekly_forecast_email.send_weekly_email")
    @patch("flows.weekly_forecast_email.generate_summary")
    @patch("flows.weekly_forecast_email.update_execution_sac")
    @patch("flows.weekly_forecast_email.update_execution_ppo")
    @patch("flows.weekly_forecast_email.get_order_history_sac")
    @patch("flows.weekly_forecast_email.get_order_history_ppo")
    @patch("flows.weekly_forecast_email.submit_orders_hrp")
    @patch("flows.weekly_forecast_email.submit_orders_sac")
    @patch("flows.weekly_forecast_email.submit_orders_ppo")
    @patch("flows.weekly_forecast_email.store_experience_sac")
    @patch("flows.weekly_forecast_email.store_experience_ppo")
    @patch("flows.weekly_forecast_email.generate_orders_hrp")
    @patch("flows.weekly_forecast_email.generate_orders_sac")
    @patch("flows.weekly_forecast_email.generate_orders_ppo")
    @patch("flows.weekly_forecast_email.allocate_hrp")
    @patch("flows.weekly_forecast_email.infer_sac")
    @patch("flows.weekly_forecast_email.infer_ppo")
    @patch("flows.weekly_forecast_email.get_patchtst_forecast")
    @patch("flows.weekly_forecast_email.get_lstm_forecast")
    @patch("flows.weekly_forecast_email.get_news_sentiment")
    @patch("flows.weekly_forecast_email.get_fundamentals")
    @patch("flows.weekly_forecast_email.get_hrp_portfolio")
    @patch("flows.weekly_forecast_email.get_sac_portfolio")
    @patch("flows.weekly_forecast_email.get_ppo_portfolio")
    @patch("flows.weekly_forecast_email.get_halal_universe")
    def test_full_flow_all_algorithms_run(
        self,
        mock_universe,
        mock_ppo_portfolio,
        mock_sac_portfolio,
        mock_hrp_portfolio,
        mock_fundamentals,
        mock_news,
        mock_lstm,
        mock_patchtst,
        mock_infer_ppo,
        mock_infer_sac,
        mock_allocate_hrp,
        mock_gen_ppo,
        mock_gen_sac,
        mock_gen_hrp,
        mock_store_ppo,
        mock_store_sac,
        mock_submit_ppo,
        mock_submit_sac,
        mock_submit_hrp,
        mock_history_ppo,
        mock_history_sac,
        mock_update_ppo,
        mock_update_sac,
        mock_summary,
        mock_email,
        mock_universe_response,
        mock_portfolio_no_open_orders,
        mock_lstm_response,
        mock_patchtst_response,
        mock_ppo_allocation,
        mock_sac_allocation,
        mock_hrp_allocation,
        mock_news_response,
        mock_fundamentals_response,
        mock_generate_orders_response,
        mock_submit_orders_response,
        mock_summary_response,
        mock_email_response,
    ):
        """Test full flow execution when all algorithms can run (no open orders)."""
        # Setup Phase 0 mocks
        mock_universe.submit.return_value.result.return_value = mock_universe_response
        mock_ppo_portfolio.submit.return_value.result.return_value = (
            mock_portfolio_no_open_orders
        )
        mock_sac_portfolio.submit.return_value.result.return_value = (
            mock_portfolio_no_open_orders
        )
        mock_hrp_portfolio.submit.return_value.result.return_value = (
            mock_portfolio_no_open_orders
        )

        # Setup Phase 1 mocks
        mock_fundamentals.submit.return_value.result.return_value = (
            mock_fundamentals_response
        )
        mock_news.submit.return_value.result.return_value = mock_news_response
        mock_lstm.submit.return_value.result.return_value = mock_lstm_response
        mock_patchtst.submit.return_value.result.return_value = mock_patchtst_response

        # Setup Phase 2 mocks
        mock_infer_ppo.submit.return_value.result.return_value = mock_ppo_allocation
        mock_infer_sac.submit.return_value.result.return_value = mock_sac_allocation
        mock_allocate_hrp.submit.return_value.result.return_value = mock_hrp_allocation

        # Setup Phase 3 mocks
        mock_gen_ppo.submit.return_value.result.return_value = (
            mock_generate_orders_response
        )
        mock_gen_sac.submit.return_value.result.return_value = (
            mock_generate_orders_response
        )
        mock_gen_hrp.submit.return_value.result.return_value = (
            mock_generate_orders_response
        )

        # Setup Phase 4 mocks
        mock_submit_ppo.submit.return_value.result.return_value = (
            mock_submit_orders_response
        )
        mock_submit_sac.submit.return_value.result.return_value = SubmitOrdersResponse(
            account="sac", orders_submitted=1, orders_failed=0, results=[]
        )
        mock_submit_hrp.submit.return_value.result.return_value = SubmitOrdersResponse(
            account="hrp", orders_submitted=2, orders_failed=0, results=[]
        )

        # Setup Phase 5 mocks
        mock_history_ppo.submit.return_value.result.return_value = []
        mock_history_sac.submit.return_value.result.return_value = []

        # Setup Phase 6 mocks
        mock_summary.return_value = mock_summary_response
        mock_email.return_value = mock_email_response

        # Run flow
        result = weekly_forecast_email_flow()

        # Verify result structure
        assert result["symbols_count"] == 20  # Top 20 from universe
        assert result["skipped_algorithms"] == []  # None skipped
        assert result["ppo"]["skipped"] is False
        assert result["sac"]["skipped"] is False
        assert result["hrp"]["skipped"] is False
        assert result["email"]["is_success"] is True

        # Verify all allocation tasks were called
        mock_infer_ppo.submit.assert_called_once()
        mock_infer_sac.submit.assert_called_once()
        mock_allocate_hrp.submit.assert_called_once()

        # Verify experience was stored for PPO and SAC
        mock_store_ppo.submit.assert_called_once()
        mock_store_sac.submit.assert_called_once()


# =============================================================================
# Skip Logic Tests
# =============================================================================


class TestSkipLogic:
    """Test algorithm skip logic when open orders exist."""

    @patch("flows.weekly_forecast_email.send_weekly_email")
    @patch("flows.weekly_forecast_email.generate_summary")
    @patch("flows.weekly_forecast_email.update_execution_sac")
    @patch("flows.weekly_forecast_email.update_execution_ppo")
    @patch("flows.weekly_forecast_email.get_order_history_sac")
    @patch("flows.weekly_forecast_email.get_order_history_ppo")
    @patch("flows.weekly_forecast_email.submit_orders_hrp")
    @patch("flows.weekly_forecast_email.submit_orders_sac")
    @patch("flows.weekly_forecast_email.submit_orders_ppo")
    @patch("flows.weekly_forecast_email.store_experience_sac")
    @patch("flows.weekly_forecast_email.store_experience_ppo")
    @patch("flows.weekly_forecast_email.generate_orders_hrp")
    @patch("flows.weekly_forecast_email.generate_orders_sac")
    @patch("flows.weekly_forecast_email.generate_orders_ppo")
    @patch("flows.weekly_forecast_email.allocate_hrp")
    @patch("flows.weekly_forecast_email.infer_sac")
    @patch("flows.weekly_forecast_email.infer_ppo")
    @patch("flows.weekly_forecast_email.get_patchtst_forecast")
    @patch("flows.weekly_forecast_email.get_lstm_forecast")
    @patch("flows.weekly_forecast_email.get_news_sentiment")
    @patch("flows.weekly_forecast_email.get_fundamentals")
    @patch("flows.weekly_forecast_email.get_hrp_portfolio")
    @patch("flows.weekly_forecast_email.get_sac_portfolio")
    @patch("flows.weekly_forecast_email.get_ppo_portfolio")
    @patch("flows.weekly_forecast_email.get_halal_universe")
    def test_skip_ppo_when_open_orders(
        self,
        mock_universe,
        mock_ppo_portfolio,
        mock_sac_portfolio,
        mock_hrp_portfolio,
        mock_fundamentals,
        mock_news,
        mock_lstm,
        mock_patchtst,
        mock_infer_ppo,
        mock_infer_sac,
        mock_allocate_hrp,
        mock_gen_ppo,
        mock_gen_sac,
        mock_gen_hrp,
        mock_store_ppo,
        mock_store_sac,
        mock_submit_ppo,
        mock_submit_sac,
        mock_submit_hrp,
        mock_history_ppo,
        mock_history_sac,
        mock_update_ppo,
        mock_update_sac,
        mock_summary,
        mock_email,
        mock_universe_response,
        mock_portfolio_no_open_orders,
        mock_portfolio_with_open_orders,
        mock_lstm_response,
        mock_patchtst_response,
        mock_sac_allocation,
        mock_hrp_allocation,
        mock_generate_orders_response,
        mock_submit_orders_response,
        mock_summary_response,
        mock_email_response,
    ):
        """Test that PPO is skipped when it has open orders."""
        # PPO has open orders, SAC and HRP don't
        mock_universe.submit.return_value.result.return_value = mock_universe_response
        mock_ppo_portfolio.submit.return_value.result.return_value = (
            mock_portfolio_with_open_orders
        )
        mock_sac_portfolio.submit.return_value.result.return_value = (
            mock_portfolio_no_open_orders
        )
        mock_hrp_portfolio.submit.return_value.result.return_value = (
            mock_portfolio_no_open_orders
        )

        # Setup Phase 1 mocks
        mock_fundamentals.submit.return_value.result.return_value = MagicMock()
        mock_news.submit.return_value.result.return_value = MagicMock()
        mock_lstm.submit.return_value.result.return_value = mock_lstm_response
        mock_patchtst.submit.return_value.result.return_value = mock_patchtst_response

        # Setup Phase 2 - only SAC and HRP should be called
        mock_infer_sac.submit.return_value.result.return_value = mock_sac_allocation
        mock_allocate_hrp.submit.return_value.result.return_value = mock_hrp_allocation

        # Setup remaining phases
        mock_gen_ppo.submit.return_value.result.return_value = SkippedOrdersResponse(
            algorithm="ppo"
        )
        mock_gen_sac.submit.return_value.result.return_value = (
            mock_generate_orders_response
        )
        mock_gen_hrp.submit.return_value.result.return_value = (
            mock_generate_orders_response
        )

        mock_submit_ppo.submit.return_value.result.return_value = SkippedSubmitResponse(
            account="ppo"
        )
        mock_submit_sac.submit.return_value.result.return_value = (
            mock_submit_orders_response
        )
        mock_submit_hrp.submit.return_value.result.return_value = (
            mock_submit_orders_response
        )

        mock_history_sac.submit.return_value.result.return_value = []

        mock_summary.return_value = mock_summary_response
        mock_email.return_value = mock_email_response

        # Run flow
        result = weekly_forecast_email_flow()

        # Verify PPO was skipped
        assert "PPO" in result["skipped_algorithms"]
        assert result["ppo"]["skipped"] is True
        assert result["sac"]["skipped"] is False
        assert result["hrp"]["skipped"] is False

        # PPO inference should NOT be called
        mock_infer_ppo.submit.assert_not_called()

        # PPO experience should NOT be stored
        mock_store_ppo.submit.assert_not_called()

        # SAC and HRP should still run
        mock_infer_sac.submit.assert_called_once()
        mock_allocate_hrp.submit.assert_called_once()

    @patch("flows.weekly_forecast_email.send_weekly_email")
    @patch("flows.weekly_forecast_email.generate_summary")
    @patch("flows.weekly_forecast_email.update_execution_sac")
    @patch("flows.weekly_forecast_email.update_execution_ppo")
    @patch("flows.weekly_forecast_email.get_order_history_sac")
    @patch("flows.weekly_forecast_email.get_order_history_ppo")
    @patch("flows.weekly_forecast_email.submit_orders_hrp")
    @patch("flows.weekly_forecast_email.submit_orders_sac")
    @patch("flows.weekly_forecast_email.submit_orders_ppo")
    @patch("flows.weekly_forecast_email.store_experience_sac")
    @patch("flows.weekly_forecast_email.store_experience_ppo")
    @patch("flows.weekly_forecast_email.generate_orders_hrp")
    @patch("flows.weekly_forecast_email.generate_orders_sac")
    @patch("flows.weekly_forecast_email.generate_orders_ppo")
    @patch("flows.weekly_forecast_email.allocate_hrp")
    @patch("flows.weekly_forecast_email.infer_sac")
    @patch("flows.weekly_forecast_email.infer_ppo")
    @patch("flows.weekly_forecast_email.get_patchtst_forecast")
    @patch("flows.weekly_forecast_email.get_lstm_forecast")
    @patch("flows.weekly_forecast_email.get_news_sentiment")
    @patch("flows.weekly_forecast_email.get_fundamentals")
    @patch("flows.weekly_forecast_email.get_hrp_portfolio")
    @patch("flows.weekly_forecast_email.get_sac_portfolio")
    @patch("flows.weekly_forecast_email.get_ppo_portfolio")
    @patch("flows.weekly_forecast_email.get_halal_universe")
    def test_skip_all_algorithms_when_all_have_open_orders(
        self,
        mock_universe,
        mock_ppo_portfolio,
        mock_sac_portfolio,
        mock_hrp_portfolio,
        mock_fundamentals,
        mock_news,
        mock_lstm,
        mock_patchtst,
        mock_infer_ppo,
        mock_infer_sac,
        mock_allocate_hrp,
        mock_gen_ppo,
        mock_gen_sac,
        mock_gen_hrp,
        mock_store_ppo,
        mock_store_sac,
        mock_submit_ppo,
        mock_submit_sac,
        mock_submit_hrp,
        mock_history_ppo,
        mock_history_sac,
        mock_update_ppo,
        mock_update_sac,
        mock_summary,
        mock_email,
        mock_universe_response,
        mock_portfolio_with_open_orders,
        mock_lstm_response,
        mock_patchtst_response,
        mock_summary_response,
        mock_email_response,
    ):
        """Test that all algorithms are skipped when all have open orders."""
        # All have open orders
        mock_universe.submit.return_value.result.return_value = mock_universe_response
        mock_ppo_portfolio.submit.return_value.result.return_value = (
            mock_portfolio_with_open_orders
        )
        mock_sac_portfolio.submit.return_value.result.return_value = (
            mock_portfolio_with_open_orders
        )
        mock_hrp_portfolio.submit.return_value.result.return_value = (
            mock_portfolio_with_open_orders
        )

        # Phase 1 still runs to generate signals/forecasts for email
        mock_fundamentals.submit.return_value.result.return_value = MagicMock()
        mock_news.submit.return_value.result.return_value = MagicMock()
        mock_lstm.submit.return_value.result.return_value = mock_lstm_response
        mock_patchtst.submit.return_value.result.return_value = mock_patchtst_response

        # All generate orders return skipped
        mock_gen_ppo.submit.return_value.result.return_value = SkippedOrdersResponse(
            algorithm="ppo"
        )
        mock_gen_sac.submit.return_value.result.return_value = SkippedOrdersResponse(
            algorithm="sac"
        )
        mock_gen_hrp.submit.return_value.result.return_value = SkippedOrdersResponse(
            algorithm="hrp"
        )

        # All submit orders return skipped
        mock_submit_ppo.submit.return_value.result.return_value = SkippedSubmitResponse(
            account="ppo"
        )
        mock_submit_sac.submit.return_value.result.return_value = SkippedSubmitResponse(
            account="sac"
        )
        mock_submit_hrp.submit.return_value.result.return_value = SkippedSubmitResponse(
            account="hrp"
        )

        mock_summary.return_value = mock_summary_response
        mock_email.return_value = mock_email_response

        # Run flow
        result = weekly_forecast_email_flow()

        # Verify all algorithms were skipped
        assert "PPO" in result["skipped_algorithms"]
        assert "SAC" in result["skipped_algorithms"]
        assert "HRP" in result["skipped_algorithms"]
        assert result["ppo"]["skipped"] is True
        assert result["sac"]["skipped"] is True
        assert result["hrp"]["skipped"] is True

        # No allocation tasks should be called
        mock_infer_ppo.submit.assert_not_called()
        mock_infer_sac.submit.assert_not_called()
        mock_allocate_hrp.submit.assert_not_called()

        # No experience should be stored
        mock_store_ppo.submit.assert_not_called()
        mock_store_sac.submit.assert_not_called()

        # Email should still be sent
        mock_email.assert_called_once()


# =============================================================================
# HRP Percentage Conversion Tests
# =============================================================================


class TestHRPPercentageConversion:
    """Test HRP percentage to decimal weight conversion."""

    def test_hrp_percentage_weights_format(self, mock_hrp_allocation):
        """Test that HRP allocation model uses percentage weights (not decimals)."""
        # HRP returns 20.0 for 20%, not 0.20
        assert mock_hrp_allocation.percentage_weights["AAPL"] == 20.0
        assert mock_hrp_allocation.percentage_weights["MSFT"] == 15.0
        assert mock_hrp_allocation.percentage_weights["GOOGL"] == 10.0

    def test_convert_percentage_to_decimal(self, mock_hrp_allocation):
        """Test conversion from percentage to decimal weights."""
        # This is what the generate_orders_hrp task does
        target_weights = {
            sym: wt / 100 for sym, wt in mock_hrp_allocation.percentage_weights.items()
        }

        assert target_weights["AAPL"] == pytest.approx(0.20, abs=0.001)
        assert target_weights["MSFT"] == pytest.approx(0.15, abs=0.001)
        assert target_weights["GOOGL"] == pytest.approx(0.10, abs=0.001)

    @patch("flows.tasks.execution.get_run_logger")
    @patch("flows.tasks.execution.get_client")
    def test_generate_orders_hrp_converts_percentages(
        self, mock_get_client, mock_get_logger
    ):
        """Test that generate_orders_hrp converts percentage to decimal weights."""
        from flows.tasks.execution import generate_orders_hrp

        # Setup mock logger
        mock_get_logger.return_value = MagicMock()

        # Setup mock client
        mock_client = MagicMock()
        mock_get_client.return_value.__enter__.return_value = mock_client
        mock_client.post.return_value.json.return_value = {
            "orders": [],
            "summary": {
                "buys": 0,
                "sells": 0,
                "total_buy_value": 0,
                "total_sell_value": 0,
                "turnover_pct": 0,
                "skipped_small_orders": 0,
            },
            "prices_used": {},
        }

        hrp_alloc = HRPAllocationResponse(
            percentage_weights={"AAPL": 50.0, "MSFT": 30.0},  # 50% and 30%
            symbols_used=2,
            symbols_excluded=[],
            as_of_date="2026-02-05",
        )

        portfolio = AlpacaPortfolioResponse(
            cash=10000.0,
            positions=[],
            open_orders_count=0,
        )

        # Run task function directly
        generate_orders_hrp.fn(hrp_alloc, portfolio, "test-run", 1)

        # Verify the API was called with decimal weights, not percentages
        call_args = mock_client.post.call_args
        json_body = call_args.kwargs["json"]

        assert json_body["target_weights"]["AAPL"] == pytest.approx(0.50, abs=0.001)
        assert json_body["target_weights"]["MSFT"] == pytest.approx(0.30, abs=0.001)


# =============================================================================
# Parallel Execution Tests
# =============================================================================


class TestParallelExecution:
    """Test that tasks are submitted in parallel where expected."""

    @patch("flows.weekly_forecast_email.send_weekly_email")
    @patch("flows.weekly_forecast_email.generate_summary")
    @patch("flows.weekly_forecast_email.update_execution_sac")
    @patch("flows.weekly_forecast_email.update_execution_ppo")
    @patch("flows.weekly_forecast_email.get_order_history_sac")
    @patch("flows.weekly_forecast_email.get_order_history_ppo")
    @patch("flows.weekly_forecast_email.submit_orders_hrp")
    @patch("flows.weekly_forecast_email.submit_orders_sac")
    @patch("flows.weekly_forecast_email.submit_orders_ppo")
    @patch("flows.weekly_forecast_email.store_experience_sac")
    @patch("flows.weekly_forecast_email.store_experience_ppo")
    @patch("flows.weekly_forecast_email.generate_orders_hrp")
    @patch("flows.weekly_forecast_email.generate_orders_sac")
    @patch("flows.weekly_forecast_email.generate_orders_ppo")
    @patch("flows.weekly_forecast_email.allocate_hrp")
    @patch("flows.weekly_forecast_email.infer_sac")
    @patch("flows.weekly_forecast_email.infer_ppo")
    @patch("flows.weekly_forecast_email.get_patchtst_forecast")
    @patch("flows.weekly_forecast_email.get_lstm_forecast")
    @patch("flows.weekly_forecast_email.get_news_sentiment")
    @patch("flows.weekly_forecast_email.get_fundamentals")
    @patch("flows.weekly_forecast_email.get_hrp_portfolio")
    @patch("flows.weekly_forecast_email.get_sac_portfolio")
    @patch("flows.weekly_forecast_email.get_ppo_portfolio")
    @patch("flows.weekly_forecast_email.get_halal_universe")
    def test_phase0_tasks_submitted_in_parallel(
        self,
        mock_universe,
        mock_ppo_portfolio,
        mock_sac_portfolio,
        mock_hrp_portfolio,
        mock_fundamentals,
        mock_news,
        mock_lstm,
        mock_patchtst,
        mock_infer_ppo,
        mock_infer_sac,
        mock_allocate_hrp,
        mock_gen_ppo,
        mock_gen_sac,
        mock_gen_hrp,
        mock_store_ppo,
        mock_store_sac,
        mock_submit_ppo,
        mock_submit_sac,
        mock_submit_hrp,
        mock_history_ppo,
        mock_history_sac,
        mock_update_ppo,
        mock_update_sac,
        mock_summary,
        mock_email,
        mock_universe_response,
        mock_portfolio_no_open_orders,
        mock_lstm_response,
        mock_patchtst_response,
        mock_ppo_allocation,
        mock_sac_allocation,
        mock_hrp_allocation,
        mock_news_response,
        mock_fundamentals_response,
        mock_generate_orders_response,
        mock_submit_orders_response,
        mock_summary_response,
        mock_email_response,
    ):
        """Test Phase 0 tasks are submitted (not called directly)."""
        # Setup all mocks (same as full flow test)
        mock_universe.submit.return_value.result.return_value = mock_universe_response
        mock_ppo_portfolio.submit.return_value.result.return_value = (
            mock_portfolio_no_open_orders
        )
        mock_sac_portfolio.submit.return_value.result.return_value = (
            mock_portfolio_no_open_orders
        )
        mock_hrp_portfolio.submit.return_value.result.return_value = (
            mock_portfolio_no_open_orders
        )
        mock_fundamentals.submit.return_value.result.return_value = (
            mock_fundamentals_response
        )
        mock_news.submit.return_value.result.return_value = mock_news_response
        mock_lstm.submit.return_value.result.return_value = mock_lstm_response
        mock_patchtst.submit.return_value.result.return_value = mock_patchtst_response
        mock_infer_ppo.submit.return_value.result.return_value = mock_ppo_allocation
        mock_infer_sac.submit.return_value.result.return_value = mock_sac_allocation
        mock_allocate_hrp.submit.return_value.result.return_value = mock_hrp_allocation
        mock_gen_ppo.submit.return_value.result.return_value = (
            mock_generate_orders_response
        )
        mock_gen_sac.submit.return_value.result.return_value = (
            mock_generate_orders_response
        )
        mock_gen_hrp.submit.return_value.result.return_value = (
            mock_generate_orders_response
        )
        mock_submit_ppo.submit.return_value.result.return_value = (
            mock_submit_orders_response
        )
        mock_submit_sac.submit.return_value.result.return_value = (
            mock_submit_orders_response
        )
        mock_submit_hrp.submit.return_value.result.return_value = (
            mock_submit_orders_response
        )
        mock_history_ppo.submit.return_value.result.return_value = []
        mock_history_sac.submit.return_value.result.return_value = []
        mock_summary.return_value = mock_summary_response
        mock_email.return_value = mock_email_response

        # Run flow
        weekly_forecast_email_flow()

        # Verify Phase 0 tasks were submitted (parallel execution)
        mock_universe.submit.assert_called_once()
        mock_ppo_portfolio.submit.assert_called_once()
        mock_sac_portfolio.submit.assert_called_once()
        mock_hrp_portfolio.submit.assert_called_once()

        # Verify Phase 1 tasks were submitted
        mock_fundamentals.submit.assert_called_once()
        mock_news.submit.assert_called_once()
        mock_lstm.submit.assert_called_once()
        mock_patchtst.submit.assert_called_once()

    def test_universe_symbols_sliced_to_top_20(self, mock_universe_response):
        """Test that only top 20 symbols are used from universe."""
        symbols = mock_universe_response.symbols[:20]
        assert len(symbols) == 20
        # First 20 symbols should be SYM0 through SYM19
        assert symbols[0] == "SYM0"
        assert symbols[19] == "SYM19"

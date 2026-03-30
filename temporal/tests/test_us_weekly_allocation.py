"""Tests for US weekly allocation Temporal workflow.

Tests:
- Full workflow execution with mocked activities
- Skip logic when algorithms have open orders
- Independent per-algorithm sell-wait-buy (sells don't block other accounts)
"""

from concurrent.futures import ThreadPoolExecutor

import pytest
from temporalio import activity
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from models import (
    ActiveSymbolsResponse,
    AlpacaPortfolioResponse,
    FundamentalsResponse,
    GenerateOrdersResponse,
    HRPAllocationResponse,
    LSTMInferenceResponse,
    NewsSignalResponse,
    OrderModel,
    OrderSummary,
    PatchTSTInferenceResponse,
    PositionModel,
    SACInferenceResponse,
    SkippedOrdersResponse,
    SkippedSubmitResponse,
    SubmitOrdersResponse,
    WeeklyReportEmailResponse,
    WeeklySummaryResponse,
)
from workflows.us_weekly_allocation import USWeeklyAllocationWorkflow


@pytest.fixture
def active_symbols():
    return ActiveSymbolsResponse(
        symbols=[f"SYM{i}" for i in range(15)],
        source_model="sac",
        model_version="v1.0.0",
    )


@pytest.fixture
def portfolio_no_open():
    return AlpacaPortfolioResponse(
        cash=10000.0,
        positions=[PositionModel(symbol="AAPL", qty=10.0, market_value=1750.0)],
        open_orders_count=0,
    )


@pytest.fixture
def portfolio_with_open():
    return AlpacaPortfolioResponse(
        cash=10000.0,
        positions=[PositionModel(symbol="AAPL", qty=10.0, market_value=1750.0)],
        open_orders_count=3,
    )


@pytest.fixture
def lstm_resp():
    return LSTMInferenceResponse(
        predictions=[
            {
                "symbol": "AAPL",
                "predicted_weekly_return_pct": 2.5,
                "direction": "up",
                "has_enough_history": True,
            },
        ],
        model_version="v1.0.0",
        as_of_date="2026-02-05",
        target_week_start="2026-02-10",
        target_week_end="2026-02-14",
    )


@pytest.fixture
def patchtst_resp():
    return PatchTSTInferenceResponse(
        predictions=[
            {
                "symbol": "AAPL",
                "predicted_weekly_return_pct": 3.0,
                "direction": "up",
                "has_enough_history": True,
            },
        ],
        model_version="v1.0.0",
        as_of_date="2026-02-05",
        signals_used=["ohlcv"],
        target_week_start="2026-02-10",
        target_week_end="2026-02-14",
    )


@pytest.fixture
def news_resp():
    return NewsSignalResponse(
        per_symbol=[{"symbol": "AAPL", "sentiment_score": 0.5, "article_count": 10}],
        as_of_date="2026-02-05",
    )


@pytest.fixture
def fundamentals_resp():
    return FundamentalsResponse(
        per_symbol=[{"symbol": "AAPL", "ratios": {"gross_margin": 0.42}}],
        as_of_date="2026-02-05",
    )


@pytest.fixture
def sac_alloc():
    return SACInferenceResponse(
        target_weights={"AAPL": 0.25, "CASH": 0.75},
        turnover=0.12,
        model_version="v1.0.0",
        target_week_start="2026-02-10",
        target_week_end="2026-02-14",
    )


@pytest.fixture
def hrp_alloc():
    return HRPAllocationResponse(
        universe="halal_filtered",
        percentage_weights={"AAPL": 20.0, "MSFT": 15.0},
        symbols_used=2,
        symbols_excluded=[],
        as_of_date="2026-02-05",
    )


@pytest.fixture
def buy_only_orders():
    return GenerateOrdersResponse(
        orders=[
            OrderModel(
                client_order_id="paper:2026-02-05:attempt-1:AAPL:BUY",
                symbol="AAPL",
                side="buy",
                qty=5.0,
                type="market",
                time_in_force="day",
            ),
        ],
        summary=OrderSummary(
            buys=1,
            sells=0,
            total_buy_value=877.50,
            total_sell_value=0,
            turnover_pct=8.8,
            skipped_small_orders=0,
            skipped_below_threshold=0,
        ),
        prices_used={"AAPL": 175.50},
    )


@pytest.fixture
def sell_and_buy_orders():
    return GenerateOrdersResponse(
        orders=[
            OrderModel(
                client_order_id="paper:2026-02-05:attempt-1:MSFT:SELL",
                symbol="MSFT",
                side="sell",
                qty=3.0,
                type="market",
                time_in_force="day",
            ),
            OrderModel(
                client_order_id="paper:2026-02-05:attempt-1:AAPL:BUY",
                symbol="AAPL",
                side="buy",
                qty=5.0,
                type="market",
                time_in_force="day",
            ),
        ],
        summary=OrderSummary(
            buys=1,
            sells=1,
            total_buy_value=877.50,
            total_sell_value=1260.00,
            turnover_pct=12.0,
            skipped_small_orders=0,
            skipped_below_threshold=0,
        ),
        prices_used={"AAPL": 175.50, "MSFT": 420.00},
    )


@pytest.fixture
def submit_resp():
    return SubmitOrdersResponse(
        account="sac", orders_submitted=1, orders_failed=0, skipped=False, results=[]
    )


@pytest.fixture
def summary_resp():
    return WeeklySummaryResponse(
        summary={"overview": "Weekly analysis."},
        provider="openai",
        model_used="gpt-4o-mini",
        tokens_used=500,
    )


@pytest.fixture
def email_resp():
    return WeeklyReportEmailResponse(
        is_success=True,
        subject="Weekly Forecast Report",
        body="<html>report</html>",
    )


def _make_mock_activities(
    active_symbols,
    sac_portfolio,
    hrp_portfolio,
    fundamentals_resp,
    news_resp,
    lstm_resp,
    patchtst_resp,
    sac_alloc,
    hrp_alloc,
    sac_orders,
    hrp_orders,
    submit_resp,
    summary_resp,
    email_resp,
    check_order_statuses_fn=None,
):
    """Build a list of mock activity functions that return fixture data."""

    @activity.defn(name="resolve_next_attempt")
    def mock_resolve_next_attempt(run_id, as_of_date) -> int:
        return 1

    @activity.defn(name="get_active_symbols")
    def mock_get_active_symbols() -> ActiveSymbolsResponse:
        return active_symbols

    @activity.defn(name="get_sac_portfolio")
    def mock_get_sac_portfolio() -> AlpacaPortfolioResponse:
        return sac_portfolio

    @activity.defn(name="get_hrp_portfolio")
    def mock_get_hrp_portfolio() -> AlpacaPortfolioResponse:
        return hrp_portfolio

    @activity.defn(name="get_fundamentals")
    def mock_get_fundamentals(symbols):
        return fundamentals_resp

    @activity.defn(name="get_news_sentiment")
    def mock_get_news_sentiment(symbols, as_of_date, run_id):
        return news_resp

    @activity.defn(name="get_lstm_forecast")
    def mock_get_lstm_forecast(as_of_date, symbols=None):
        return lstm_resp

    @activity.defn(name="get_patchtst_forecast")
    def mock_get_patchtst_forecast(as_of_date, symbols=None):
        return patchtst_resp

    @activity.defn(name="infer_sac")
    def mock_infer_sac(portfolio, as_of_date):
        return sac_alloc

    @activity.defn(name="allocate_hrp")
    def mock_allocate_hrp(as_of_date, universe="halal_filtered"):
        return hrp_alloc

    @activity.defn(name="generate_orders_sac")
    def mock_generate_orders_sac(allocation, portfolio, run_id, attempt):
        return sac_orders

    @activity.defn(name="generate_orders_hrp")
    def mock_generate_orders_hrp(allocation, portfolio, run_id, attempt):
        return hrp_orders

    @activity.defn(name="store_experience_sac")
    def mock_store_experience_sac(*args):
        return None

    @activity.defn(name="submit_orders_sac")
    def mock_submit_orders_sac(orders):
        if isinstance(orders, SkippedOrdersResponse) or getattr(
            orders, "skipped", False
        ):
            return SkippedSubmitResponse(account="sac")
        return submit_resp

    @activity.defn(name="submit_orders_hrp")
    def mock_submit_orders_hrp(orders):
        if isinstance(orders, SkippedOrdersResponse) or getattr(
            orders, "skipped", False
        ):
            return SkippedSubmitResponse(account="hrp")
        return submit_resp

    @activity.defn(name="check_order_statuses")
    def mock_check_order_statuses(account, client_order_ids):
        if check_order_statuses_fn is not None:
            return check_order_statuses_fn(account, client_order_ids)
        return []

    @activity.defn(name="get_order_history_sac")
    def mock_get_order_history_sac(after_date):
        return []

    @activity.defn(name="update_execution_sac")
    def mock_update_execution_sac(run_id, orders, history):
        return None

    @activity.defn(name="generate_summary")
    def mock_generate_summary(*args):
        return summary_resp

    @activity.defn(name="send_weekly_email")
    def mock_send_weekly_email(*args):
        return email_resp

    return [
        mock_resolve_next_attempt,
        mock_get_active_symbols,
        mock_get_sac_portfolio,
        mock_get_hrp_portfolio,
        mock_get_fundamentals,
        mock_get_news_sentiment,
        mock_get_lstm_forecast,
        mock_get_patchtst_forecast,
        mock_infer_sac,
        mock_allocate_hrp,
        mock_generate_orders_sac,
        mock_generate_orders_hrp,
        mock_store_experience_sac,
        mock_submit_orders_sac,
        mock_submit_orders_hrp,
        mock_check_order_statuses,
        mock_get_order_history_sac,
        mock_update_execution_sac,
        mock_generate_summary,
        mock_send_weekly_email,
    ]


class TestUSWeeklyAllocationAllRun:
    """Test full flow with no open orders -- all algorithms run."""

    @pytest.mark.asyncio
    async def test_full_flow_all_algorithms_run(
        self,
        active_symbols,
        portfolio_no_open,
        lstm_resp,
        patchtst_resp,
        news_resp,
        fundamentals_resp,
        sac_alloc,
        hrp_alloc,
        buy_only_orders,
        submit_resp,
        summary_resp,
        email_resp,
    ):
        activities = _make_mock_activities(
            active_symbols=active_symbols,
            sac_portfolio=portfolio_no_open,
            hrp_portfolio=portfolio_no_open,
            fundamentals_resp=fundamentals_resp,
            news_resp=news_resp,
            lstm_resp=lstm_resp,
            patchtst_resp=patchtst_resp,
            sac_alloc=sac_alloc,
            hrp_alloc=hrp_alloc,
            sac_orders=buy_only_orders,
            hrp_orders=buy_only_orders,
            submit_resp=submit_resp,
            summary_resp=summary_resp,
            email_resp=email_resp,
        )

        async with await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter
        ) as env:
            async with Worker(
                env.client,
                task_queue="test-queue",
                workflows=[USWeeklyAllocationWorkflow],
                activities=activities,
                activity_executor=ThreadPoolExecutor(),
            ):
                result = await env.client.execute_workflow(
                    USWeeklyAllocationWorkflow.run,
                    id="test-us-inference",
                    task_queue="test-queue",
                )

            assert result["symbols_count"] == 15
            assert result["skipped_algorithms"] == []
            assert result["sac"]["skipped"] is False
            assert result["hrp"]["skipped"] is False
            assert result["email"]["is_success"] is True


class TestUSWeeklyAllocationSkipLogic:
    """Test algorithm skip logic when open orders exist."""

    @pytest.mark.asyncio
    async def test_skip_sac_when_open_orders(
        self,
        active_symbols,
        portfolio_no_open,
        portfolio_with_open,
        lstm_resp,
        patchtst_resp,
        news_resp,
        fundamentals_resp,
        sac_alloc,
        hrp_alloc,
        buy_only_orders,
        submit_resp,
        summary_resp,
        email_resp,
    ):
        activities = _make_mock_activities(
            active_symbols=active_symbols,
            sac_portfolio=portfolio_with_open,  # SAC has open orders
            hrp_portfolio=portfolio_no_open,
            fundamentals_resp=fundamentals_resp,
            news_resp=news_resp,
            lstm_resp=lstm_resp,
            patchtst_resp=patchtst_resp,
            sac_alloc=sac_alloc,
            hrp_alloc=hrp_alloc,
            sac_orders=SkippedOrdersResponse(skipped=True, algorithm="sac"),
            hrp_orders=buy_only_orders,
            submit_resp=submit_resp,
            summary_resp=summary_resp,
            email_resp=email_resp,
        )

        async with await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter
        ) as env:
            async with Worker(
                env.client,
                task_queue="test-queue",
                workflows=[USWeeklyAllocationWorkflow],
                activities=activities,
                activity_executor=ThreadPoolExecutor(),
            ):
                result = await env.client.execute_workflow(
                    USWeeklyAllocationWorkflow.run,
                    id="test-us-inference-skip",
                    task_queue="test-queue",
                )

            assert "SAC" in result["skipped_algorithms"]
            assert result["sac"]["skipped"] is True
            assert result["hrp"]["skipped"] is False

    @pytest.mark.asyncio
    async def test_skip_all_when_all_have_open_orders(
        self,
        active_symbols,
        portfolio_with_open,
        lstm_resp,
        patchtst_resp,
        news_resp,
        fundamentals_resp,
        summary_resp,
        email_resp,
    ):
        activities = _make_mock_activities(
            active_symbols=active_symbols,
            sac_portfolio=portfolio_with_open,
            hrp_portfolio=portfolio_with_open,
            fundamentals_resp=fundamentals_resp,
            news_resp=news_resp,
            lstm_resp=lstm_resp,
            patchtst_resp=patchtst_resp,
            sac_alloc=None,
            hrp_alloc=None,
            sac_orders=SkippedOrdersResponse(skipped=True, algorithm="sac"),
            hrp_orders=SkippedOrdersResponse(skipped=True, algorithm="hrp"),
            submit_resp=SubmitOrdersResponse(
                account="test",
                orders_submitted=0,
                orders_failed=0,
                skipped=True,
                results=[],
            ),
            summary_resp=summary_resp,
            email_resp=email_resp,
        )

        async with await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter
        ) as env:
            async with Worker(
                env.client,
                task_queue="test-queue",
                workflows=[USWeeklyAllocationWorkflow],
                activities=activities,
                activity_executor=ThreadPoolExecutor(),
            ):
                result = await env.client.execute_workflow(
                    USWeeklyAllocationWorkflow.run,
                    id="test-us-inference-skip-all",
                    task_queue="test-queue",
                )

            assert "SAC" in result["skipped_algorithms"]
            assert "HRP" in result["skipped_algorithms"]
            assert result["sac"]["skipped"] is True
            assert result["hrp"]["skipped"] is True
            assert result["email"]["is_success"] is True


class TestUSWeeklyAllocationSellWaitBuy:
    """Test independent per-algorithm sell-wait-buy pipelines."""

    @pytest.mark.asyncio
    async def test_staggered_sell_fills_complete_independently(
        self,
        active_symbols,
        portfolio_no_open,
        lstm_resp,
        patchtst_resp,
        news_resp,
        fundamentals_resp,
        sac_alloc,
        hrp_alloc,
        sell_and_buy_orders,
        buy_only_orders,
        submit_resp,
        summary_resp,
        email_resp,
    ):
        """SAC sells fill immediately, HRP sells need one poll cycle.

        Verifies each algorithm's sell-wait-buy runs independently -- SAC
        buys proceed without waiting for HRP sells to fill.
        """
        hrp_poll_count = {"n": 0}

        def staggered_check(account, client_order_ids):
            if account == "sac":
                return [
                    {"client_order_id": cid, "status": "filled"}
                    for cid in client_order_ids
                ]
            if account == "hrp":
                hrp_poll_count["n"] += 1
                if hrp_poll_count["n"] <= 1:
                    return [
                        {"client_order_id": cid, "status": "pending_new"}
                        for cid in client_order_ids
                    ]
                return [
                    {"client_order_id": cid, "status": "filled"}
                    for cid in client_order_ids
                ]
            return []

        activities = _make_mock_activities(
            active_symbols=active_symbols,
            sac_portfolio=portfolio_no_open,
            hrp_portfolio=portfolio_no_open,
            fundamentals_resp=fundamentals_resp,
            news_resp=news_resp,
            lstm_resp=lstm_resp,
            patchtst_resp=patchtst_resp,
            sac_alloc=sac_alloc,
            hrp_alloc=hrp_alloc,
            sac_orders=sell_and_buy_orders,
            hrp_orders=sell_and_buy_orders,
            submit_resp=submit_resp,
            summary_resp=summary_resp,
            email_resp=email_resp,
            check_order_statuses_fn=staggered_check,
        )

        async with await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter
        ) as env:
            async with Worker(
                env.client,
                task_queue="test-queue",
                workflows=[USWeeklyAllocationWorkflow],
                activities=activities,
                activity_executor=ThreadPoolExecutor(),
            ):
                result = await env.client.execute_workflow(
                    USWeeklyAllocationWorkflow.run,
                    id="test-us-staggered-sells",
                    task_queue="test-queue",
                )

            assert result["skipped_algorithms"] == []
            assert result["sac"]["skipped"] is False
            assert result["hrp"]["skipped"] is False
            assert result["sac"]["orders_submitted"] > 0
            assert result["hrp"]["orders_submitted"] > 0
            assert result["email"]["is_success"] is True
            assert hrp_poll_count["n"] == 2

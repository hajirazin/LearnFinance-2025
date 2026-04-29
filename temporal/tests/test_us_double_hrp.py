"""Tests for US Double HRP Temporal workflow.

Covers:
- Happy path: stage1 + sticky + stage2 + record_final + sell-wait-buy + email
- Skip path: open orders on dhrp account -> no HRP/sticky/orders, email skipped=True
- Sell-wait-buy interaction: terminal sells -> single poll, buys submitted second
- Attempt isolation: resolve_next_attempt called with accounts=['dhrp']
"""

from concurrent.futures import ThreadPoolExecutor

import pytest
from temporalio import activity
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from models import (
    AlpacaPortfolioResponse,
    GenerateOrdersResponse,
    HRPAllocationResponse,
    OrderModel,
    OrderSummary,
    PositionModel,
    RecordFinalWeightsResponse,
    SkippedOrdersResponse,
    SkippedSubmitResponse,
    StickyTopNResponse,
    SubmitOrdersResponse,
    WeeklyReportEmailResponse,
    WeeklySummaryResponse,
)
from workflows.us_double_hrp import USDoubleHRPWorkflow


@pytest.fixture
def universe_data():
    """Simulate a halal_new universe response (~410 in production)."""
    return {
        "stocks": [{"symbol": f"SYM{i}"} for i in range(20)],
        "total_stocks": 20,
        "source": "halal_new",
    }


@pytest.fixture
def dhrp_portfolio_no_open():
    return AlpacaPortfolioResponse(
        cash=10000.0,
        positions=[PositionModel(symbol="AAPL", qty=10.0, market_value=1750.0)],
        open_orders_count=0,
    )


@pytest.fixture
def dhrp_portfolio_with_open():
    return AlpacaPortfolioResponse(
        cash=10000.0,
        positions=[PositionModel(symbol="AAPL", qty=10.0, market_value=1750.0)],
        open_orders_count=2,
    )


@pytest.fixture
def stage1_alloc():
    weights = {f"SYM{i}": round(20.0 - i * 0.8, 2) for i in range(20)}
    return HRPAllocationResponse(
        percentage_weights=weights,
        symbols_used=20,
        symbols_excluded=[],
        lookback_days=756,
        as_of_date="2026-04-28",
    )


@pytest.fixture
def stage2_alloc():
    weights = {f"SYM{i}": round(100.0 / 15, 2) for i in range(15)}
    return HRPAllocationResponse(
        percentage_weights=weights,
        symbols_used=15,
        symbols_excluded=[],
        lookback_days=252,
        as_of_date="2026-04-28",
    )


@pytest.fixture
def sticky_resp():
    selected = [f"SYM{i}" for i in range(15)]
    return StickyTopNResponse(
        selected=selected,
        reasons={s: "top_rank" for s in selected},
        kept_count=0,
        fillers_count=15,
        evicted_from_previous={},
        previous_year_week_used=None,
        universe="halal_new",
        year_week="202618",
    )


@pytest.fixture
def record_final_resp():
    return RecordFinalWeightsResponse(
        rows_updated=15, universe="halal_new", year_week="202618"
    )


@pytest.fixture
def dhrp_orders_buys_only():
    return GenerateOrdersResponse(
        orders=[
            OrderModel(
                client_order_id="paper:2026-04-28:attempt-1:SYM0:BUY",
                symbol="SYM0",
                side="buy",
                qty=5.0,
                type="market",
                time_in_force="day",
            ),
        ],
        summary=OrderSummary(
            buys=1,
            sells=0,
            total_buy_value=500.0,
            total_sell_value=0,
            turnover_pct=5.0,
            skipped_small_orders=0,
            skipped_below_threshold=0,
        ),
        prices_used={"SYM0": 100.0},
    )


@pytest.fixture
def dhrp_orders_sell_and_buy():
    return GenerateOrdersResponse(
        orders=[
            OrderModel(
                client_order_id="paper:2026-04-28:attempt-1:AAPL:SELL",
                symbol="AAPL",
                side="sell",
                qty=10.0,
                type="market",
                time_in_force="day",
            ),
            OrderModel(
                client_order_id="paper:2026-04-28:attempt-1:SYM0:BUY",
                symbol="SYM0",
                side="buy",
                qty=5.0,
                type="market",
                time_in_force="day",
            ),
        ],
        summary=OrderSummary(
            buys=1,
            sells=1,
            total_buy_value=500.0,
            total_sell_value=1750.0,
            turnover_pct=12.0,
            skipped_small_orders=0,
            skipped_below_threshold=0,
        ),
        prices_used={"AAPL": 175.0, "SYM0": 100.0},
    )


@pytest.fixture
def submit_resp():
    return SubmitOrdersResponse(
        account="dhrp",
        orders_submitted=1,
        orders_failed=0,
        skipped=False,
        results=[],
    )


@pytest.fixture
def summary_resp():
    return WeeklySummaryResponse(
        summary={"para_1_screening_overview": "Sticky retention 0/15 fillers."},
        provider="openai",
        model_used="gpt-4o-mini",
        tokens_used=350,
    )


@pytest.fixture
def email_resp():
    return WeeklyReportEmailResponse(
        is_success=True,
        subject="US Double HRP Portfolio Analysis (2026-04-28 -> 2026-05-02)",
        body="<html>us double hrp report</html>",
    )


def _make_us_double_hrp_activities(
    *,
    universe_data,
    dhrp_portfolio,
    stage1,
    stage2,
    sticky,
    record_final,
    orders,
    submit_resp,
    summary,
    email,
    resolve_calls=None,
    hrp_calls=None,
    sticky_calls=None,
    record_final_calls=None,
    submit_calls=None,
    check_order_statuses_fn=None,
):
    """Build mock activity functions for USDoubleHRPWorkflow."""

    @activity.defn(name="resolve_next_attempt")
    def mock_resolve_next_attempt(run_id, as_of_date, accounts=None) -> int:
        if resolve_calls is not None:
            resolve_calls.append(
                {"run_id": run_id, "as_of_date": as_of_date, "accounts": accounts}
            )
        return 1

    @activity.defn(name="fetch_halal_new_universe")
    def mock_fetch_universe() -> dict:
        return universe_data

    @activity.defn(name="get_dhrp_portfolio")
    def mock_get_dhrp_portfolio() -> AlpacaPortfolioResponse:
        return dhrp_portfolio

    @activity.defn(name="allocate_hrp")
    def mock_allocate_hrp(symbols, as_of_date, lookback_days=252):
        if hrp_calls is not None:
            hrp_calls.append(
                {
                    "symbols": symbols,
                    "as_of_date": as_of_date,
                    "lookback_days": lookback_days,
                }
            )
        if lookback_days == 756:
            return stage1
        return stage2

    @activity.defn(name="select_sticky_top_n")
    def mock_select_sticky(
        stage1_arg,
        universe,
        year_week,
        as_of_date,
        run_id,
        top_n,
        threshold,
    ):
        if sticky_calls is not None:
            sticky_calls.append(
                {
                    "universe": universe,
                    "year_week": year_week,
                    "as_of_date": as_of_date,
                    "run_id": run_id,
                    "top_n": top_n,
                    "threshold": threshold,
                }
            )
        return sticky

    @activity.defn(name="record_final_weights")
    def mock_record_final(universe, year_week, final_weights_pct):
        if record_final_calls is not None:
            record_final_calls.append(
                {
                    "universe": universe,
                    "year_week": year_week,
                    "n_weights": len(final_weights_pct),
                }
            )
        return record_final

    @activity.defn(name="generate_orders_dhrp")
    def mock_generate_orders_dhrp(allocation, portfolio, run_id, attempt):
        return orders

    @activity.defn(name="submit_orders_dhrp")
    def mock_submit_orders_dhrp(orders_resp):
        if submit_calls is not None:
            submit_calls.append(orders_resp)
        if isinstance(orders_resp, SkippedOrdersResponse) or getattr(
            orders_resp, "skipped", False
        ):
            return SkippedSubmitResponse(account="dhrp")
        return submit_resp

    @activity.defn(name="check_order_statuses")
    def mock_check_order_statuses(account, client_order_ids):
        if check_order_statuses_fn is not None:
            return check_order_statuses_fn(account, client_order_ids)
        return [
            {"client_order_id": cid, "status": "filled"} for cid in client_order_ids
        ]

    @activity.defn(name="generate_us_double_hrp_summary")
    def mock_generate_us_double_hrp_summary(s1, s2, universe, top_n):
        return summary

    @activity.defn(name="send_us_double_hrp_email")
    def mock_send_us_double_hrp_email(*args, **kwargs):
        return email

    return [
        mock_resolve_next_attempt,
        mock_fetch_universe,
        mock_get_dhrp_portfolio,
        mock_allocate_hrp,
        mock_select_sticky,
        mock_record_final,
        mock_generate_orders_dhrp,
        mock_submit_orders_dhrp,
        mock_check_order_statuses,
        mock_generate_us_double_hrp_summary,
        mock_send_us_double_hrp_email,
    ]


class TestUSDoubleHRPHappyPath:
    @pytest.mark.asyncio
    async def test_full_workflow_happy_path(
        self,
        universe_data,
        dhrp_portfolio_no_open,
        stage1_alloc,
        stage2_alloc,
        sticky_resp,
        record_final_resp,
        dhrp_orders_buys_only,
        submit_resp,
        summary_resp,
        email_resp,
    ):
        hrp_calls: list[dict] = []
        sticky_calls: list[dict] = []
        record_final_calls: list[dict] = []

        activities = _make_us_double_hrp_activities(
            universe_data=universe_data,
            dhrp_portfolio=dhrp_portfolio_no_open,
            stage1=stage1_alloc,
            stage2=stage2_alloc,
            sticky=sticky_resp,
            record_final=record_final_resp,
            orders=dhrp_orders_buys_only,
            submit_resp=submit_resp,
            summary=summary_resp,
            email=email_resp,
            hrp_calls=hrp_calls,
            sticky_calls=sticky_calls,
            record_final_calls=record_final_calls,
        )

        async with (
            await WorkflowEnvironment.start_time_skipping(
                data_converter=pydantic_data_converter
            ) as env,
            Worker(
                env.client,
                task_queue="test-queue",
                workflows=[USDoubleHRPWorkflow],
                activities=activities,
                activity_executor=ThreadPoolExecutor(),
            ),
        ):
            result = await env.client.execute_workflow(
                USDoubleHRPWorkflow.run,
                id="test-us-double-hrp-happy",
                task_queue="test-queue",
            )

        assert result["skipped"] is False
        assert result["universe_symbols"] == 20
        assert result["stage1_symbols_used"] == 20
        assert result["stage2_symbols_used"] == 15
        assert result["top_n"] == 15
        assert len(result["selected_symbols"]) == 15
        assert result["kept_count"] == 0
        assert result["fillers_count"] == 15
        assert result["email"]["is_success"] is True

        assert len(hrp_calls) == 2
        assert hrp_calls[0]["lookback_days"] == 756
        assert hrp_calls[0]["symbols"] == [s["symbol"] for s in universe_data["stocks"]]
        assert hrp_calls[1]["lookback_days"] == 252
        assert hrp_calls[1]["symbols"] == sticky_resp.selected

        assert len(sticky_calls) == 1
        assert sticky_calls[0]["universe"] == "halal_new"
        assert sticky_calls[0]["top_n"] == 15
        assert sticky_calls[0]["threshold"] == 1.0
        assert len(sticky_calls[0]["year_week"]) == 6

        assert len(record_final_calls) == 1
        assert record_final_calls[0]["universe"] == "halal_new"
        assert record_final_calls[0]["n_weights"] == 15
        assert record_final_calls[0]["year_week"] == sticky_calls[0]["year_week"], (
            "record_final_weights must use the same year_week as sticky selection"
        )


class TestUSDoubleHRPSkipPath:
    @pytest.mark.asyncio
    async def test_skip_when_open_orders(
        self,
        universe_data,
        dhrp_portfolio_with_open,
        stage1_alloc,
        stage2_alloc,
        sticky_resp,
        record_final_resp,
        dhrp_orders_buys_only,
        submit_resp,
        summary_resp,
        email_resp,
    ):
        hrp_calls: list[dict] = []
        sticky_calls: list[dict] = []
        record_final_calls: list[dict] = []
        submit_calls: list = []

        activities = _make_us_double_hrp_activities(
            universe_data=universe_data,
            dhrp_portfolio=dhrp_portfolio_with_open,
            stage1=stage1_alloc,
            stage2=stage2_alloc,
            sticky=sticky_resp,
            record_final=record_final_resp,
            orders=dhrp_orders_buys_only,
            submit_resp=submit_resp,
            summary=summary_resp,
            email=email_resp,
            hrp_calls=hrp_calls,
            sticky_calls=sticky_calls,
            record_final_calls=record_final_calls,
            submit_calls=submit_calls,
        )

        async with (
            await WorkflowEnvironment.start_time_skipping(
                data_converter=pydantic_data_converter
            ) as env,
            Worker(
                env.client,
                task_queue="test-queue",
                workflows=[USDoubleHRPWorkflow],
                activities=activities,
                activity_executor=ThreadPoolExecutor(),
            ),
        ):
            result = await env.client.execute_workflow(
                USDoubleHRPWorkflow.run,
                id="test-us-double-hrp-skip",
                task_queue="test-queue",
            )

        assert result["skipped"] is True
        assert result["skip_reason"] == "open_orders"
        assert result["email"]["is_success"] is True

        # No allocation/sticky/orders should have been triggered.
        assert hrp_calls == []
        assert sticky_calls == []
        assert record_final_calls == []
        assert submit_calls == []


class TestUSDoubleHRPSellWaitBuy:
    @pytest.mark.asyncio
    async def test_terminal_sells_then_buys(
        self,
        universe_data,
        dhrp_portfolio_no_open,
        stage1_alloc,
        stage2_alloc,
        sticky_resp,
        record_final_resp,
        dhrp_orders_sell_and_buy,
        submit_resp,
        summary_resp,
        email_resp,
    ):
        submit_calls: list = []
        status_call_count = {"n": 0}

        def status_fn(account, client_order_ids):
            status_call_count["n"] += 1
            return [
                {"client_order_id": cid, "status": "filled"} for cid in client_order_ids
            ]

        activities = _make_us_double_hrp_activities(
            universe_data=universe_data,
            dhrp_portfolio=dhrp_portfolio_no_open,
            stage1=stage1_alloc,
            stage2=stage2_alloc,
            sticky=sticky_resp,
            record_final=record_final_resp,
            orders=dhrp_orders_sell_and_buy,
            submit_resp=submit_resp,
            summary=summary_resp,
            email=email_resp,
            submit_calls=submit_calls,
            check_order_statuses_fn=status_fn,
        )

        async with (
            await WorkflowEnvironment.start_time_skipping(
                data_converter=pydantic_data_converter
            ) as env,
            Worker(
                env.client,
                task_queue="test-queue",
                workflows=[USDoubleHRPWorkflow],
                activities=activities,
                activity_executor=ThreadPoolExecutor(),
            ),
        ):
            result = await env.client.execute_workflow(
                USDoubleHRPWorkflow.run,
                id="test-us-double-hrp-sellwaitbuy",
                task_queue="test-queue",
            )

        assert result["skipped"] is False
        # submit_orders_dhrp called twice: sells first, then buys.
        assert len(submit_calls) == 2

        def _orders_of(call):
            # The activity receives dict-typed payloads (no type hints), but
            # GenerateOrdersResponse.orders carries OrderModel instances when
            # type-annotated. Support both for robustness.
            if hasattr(call, "orders"):
                return [
                    o.side if hasattr(o, "side") else o["side"] for o in call.orders
                ]
            return [o["side"] for o in call["orders"]]

        first_sides = set(_orders_of(submit_calls[0]))
        second_sides = set(_orders_of(submit_calls[1]))
        assert first_sides == {"sell"}
        assert second_sides == {"buy"}
        # Single status poll is enough since first response is already terminal.
        assert status_call_count["n"] == 1


class TestUSDoubleHRPAttemptIsolation:
    @pytest.mark.asyncio
    async def test_resolve_next_attempt_called_with_dhrp_account(
        self,
        universe_data,
        dhrp_portfolio_no_open,
        stage1_alloc,
        stage2_alloc,
        sticky_resp,
        record_final_resp,
        dhrp_orders_buys_only,
        submit_resp,
        summary_resp,
        email_resp,
    ):
        resolve_calls: list[dict] = []

        activities = _make_us_double_hrp_activities(
            universe_data=universe_data,
            dhrp_portfolio=dhrp_portfolio_no_open,
            stage1=stage1_alloc,
            stage2=stage2_alloc,
            sticky=sticky_resp,
            record_final=record_final_resp,
            orders=dhrp_orders_buys_only,
            submit_resp=submit_resp,
            summary=summary_resp,
            email=email_resp,
            resolve_calls=resolve_calls,
        )

        async with (
            await WorkflowEnvironment.start_time_skipping(
                data_converter=pydantic_data_converter
            ) as env,
            Worker(
                env.client,
                task_queue="test-queue",
                workflows=[USDoubleHRPWorkflow],
                activities=activities,
                activity_executor=ThreadPoolExecutor(),
            ),
        ):
            await env.client.execute_workflow(
                USDoubleHRPWorkflow.run,
                id="test-us-double-hrp-attempt-isolation",
                task_queue="test-queue",
            )

        assert len(resolve_calls) == 1
        assert resolve_calls[0]["accounts"] == ["dhrp"], (
            "USDoubleHRPWorkflow must scope attempt resolution to its own "
            "Alpaca account; otherwise it would collide with sac/hrp orders."
        )

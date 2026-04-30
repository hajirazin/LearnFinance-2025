"""Shared pytest fixtures for Temporal workflow tests.

Holds Pydantic-response fixtures consumed by the US Alpha-HRP and the
SAC-only ``USWeeklyAllocationWorkflow`` scenario files. Activity-mock
factories and the worker-environment helper live in ``tests/harness/``
to keep this file under the 600-line policy limit.
"""

from __future__ import annotations

import pytest

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
    PatchTSTBatchScores,
    PatchTSTInferenceResponse,
    PositionModel,
    RankBandTopNResponse,
    RecordFinalWeightsResponse,
    SACInferenceResponse,
    SubmitOrdersResponse,
    WeeklyReportEmailResponse,
    WeeklySummaryResponse,
)

# ---------------------------------------------------------------------------
# US Alpha-HRP fixtures (test_us_alpha_hrp_*)
# ---------------------------------------------------------------------------


@pytest.fixture
def universe_data():
    """Simulated halal_new universe response (~410 in production)."""
    return {
        "stocks": [{"symbol": f"SYM{i}"} for i in range(20)],
        "total_stocks": 20,
        "source": "halal_new",
    }


@pytest.fixture
def hrp_portfolio_no_open():
    return AlpacaPortfolioResponse(
        cash=10000.0,
        positions=[PositionModel(symbol="AAPL", qty=10.0, market_value=1750.0)],
        open_orders_count=0,
    )


@pytest.fixture
def hrp_portfolio_with_open():
    return AlpacaPortfolioResponse(
        cash=10000.0,
        positions=[PositionModel(symbol="AAPL", qty=10.0, market_value=1750.0)],
        open_orders_count=2,
    )


@pytest.fixture
def patchtst_scores():
    """20 valid PatchTST scores spread evenly so ranks are unambiguous."""
    return PatchTSTBatchScores(
        scores={f"SYM{i}": float(20 - i) for i in range(20)},
        model_version="v2026-04-26-abc",
        as_of_date="2026-04-28",
        target_week_start="2026-04-27",
        target_week_end="2026-05-01",
        requested_count=20,
        predicted_count=20,
        excluded_symbols=[],
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
def sticky_cold_start():
    selected = [f"SYM{i}" for i in range(15)]
    return RankBandTopNResponse(
        selected=selected,
        reasons={s: "top_rank" for s in selected},
        kept_count=0,
        fillers_count=15,
        evicted_from_previous={},
        previous_year_week_used=None,
        universe="halal_new_alpha",
        year_week="202618",
        top_n=15,
        hold_threshold=30,
    )


@pytest.fixture
def sticky_stable():
    selected = [f"SYM{i}" for i in range(15)]
    return RankBandTopNResponse(
        selected=selected,
        reasons={s: "sticky" for s in selected},
        kept_count=15,
        fillers_count=0,
        evicted_from_previous={},
        previous_year_week_used="202617",
        universe="halal_new_alpha",
        year_week="202618",
        top_n=15,
        hold_threshold=30,
    )


@pytest.fixture
def sticky_eviction():
    """13 kept, 2 evicted by rank, 2 fillers."""
    kept = [f"SYM{i}" for i in range(13)]
    fillers = ["SYM13", "SYM14"]
    selected = kept + fillers
    reasons = {s: "sticky" for s in kept}
    reasons.update({s: "top_rank" for s in fillers})
    return RankBandTopNResponse(
        selected=selected,
        reasons=reasons,
        kept_count=13,
        fillers_count=2,
        evicted_from_previous={"OLD1": "rank_out_of_hold", "OLD2": "rank_out_of_hold"},
        previous_year_week_used="202617",
        universe="halal_new_alpha",
        year_week="202618",
        top_n=15,
        hold_threshold=30,
    )


@pytest.fixture
def record_final_resp():
    return RecordFinalWeightsResponse(
        rows_updated=15, universe="halal_new_alpha", year_week="202618"
    )


@pytest.fixture
def alpha_orders_buys_only():
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
def alpha_orders_sell_and_buy():
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
        account="hrp",
        orders_submitted=1,
        orders_failed=0,
        skipped=False,
        results=[],
    )


@pytest.fixture
def summary_resp():
    return WeeklySummaryResponse(
        summary={"para_1_market_outlook": "Top names look strong."},
        provider="openai",
        model_used="gpt-4o-mini",
        tokens_used=350,
    )


@pytest.fixture
def email_resp():
    return WeeklyReportEmailResponse(
        is_success=True,
        subject="US Alpha-HRP Portfolio Analysis (2026-04-28 -> 2026-05-02)",
        body="<html>us alpha hrp report</html>",
    )


# ---------------------------------------------------------------------------
# US weekly allocation (SAC-only) fixtures (test_us_weekly_allocation_*)
# ---------------------------------------------------------------------------


@pytest.fixture
def active_symbols():
    return ActiveSymbolsResponse(
        symbols=[f"SYM{i}" for i in range(15)],
        source_model="sac",
        model_version="v1.0.0",
    )


@pytest.fixture
def sac_portfolio_no_open():
    return AlpacaPortfolioResponse(
        cash=10000.0,
        positions=[PositionModel(symbol="AAPL", qty=10.0, market_value=1750.0)],
        open_orders_count=0,
    )


@pytest.fixture
def sac_portfolio_with_open():
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
def sac_submit_resp():
    return SubmitOrdersResponse(
        account="sac", orders_submitted=1, orders_failed=0, skipped=False, results=[]
    )


@pytest.fixture
def sac_summary_resp():
    return WeeklySummaryResponse(
        summary={"overview": "Weekly analysis."},
        provider="openai",
        model_used="gpt-4o-mini",
        tokens_used=500,
    )


@pytest.fixture
def sac_email_resp():
    return WeeklyReportEmailResponse(
        is_success=True,
        subject="Weekly Forecast Report",
        body="<html>report</html>",
    )

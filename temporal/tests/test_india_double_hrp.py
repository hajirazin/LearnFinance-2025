"""Tests for India Double HRP Temporal workflow.

Covers:

* Happy path: stage1 + sticky + stage2 + record_final + LLM + email.
* Stage 1 uses 756d lookback over the full Nifty Shariah 500 universe;
  Stage 2 uses 252d lookback over ``sticky.selected`` (NOT the raw
  top-N by Stage 1 weight -- that would break the Stage 2 / sticky
  consistency invariant).
* Sticky activity is called against the ``halal_india_double_hrp``
  partition (NOT ``halal_india_alpha`` and NOT ``halal_new`` -- both
  would corrupt sister strategies' carry-sets per
  ``brain_api.core.strategy_partitions``).
* ``record_final_weights`` reuses the same ``year_week`` as the sticky
  call (rerun safety) and uses the same partition.
* Cold-start (sticky returns ``previous_year_week_used=None``) and
  steady-state (sticky returns prior week + non-empty ``kept_count``)
  both flow end-to-end without divergence.
* ``run_id`` is the default ``paper:YYYY-MM-DD`` form (India does not
  use the per-strategy ``paper:<universe>:...`` variant because it has
  no Alpaca account to dedup against).
"""

from datetime import datetime

import pytest
from temporalio import activity

from models import (
    HRPAllocationResponse,
    RecordFinalWeightsResponse,
    StickyTopNResponse,
    WeeklyReportEmailResponse,
    WeeklySummaryResponse,
)
from tests.harness.worker import worker_with_activities
from workflows.india_double_hrp import IndiaDoubleHRPWorkflow


@pytest.fixture
def mock_universe_data():
    """~210 Nifty Shariah 500 symbols (simulated with 20)."""
    return {
        "stocks": [{"symbol": f"SYM{i}.NS"} for i in range(20)],
        "total_stocks": 20,
        "source": "nifty_500_shariah",
    }


@pytest.fixture
def mock_stage1():
    """Stage 1: full universe HRP result (sorted descending by weight)."""
    weights = {f"SYM{i}.NS": round(20.0 - i * 0.8, 2) for i in range(20)}
    return HRPAllocationResponse(
        percentage_weights=weights,
        symbols_used=20,
        symbols_excluded=[],
        lookback_days=756,
        as_of_date="2026-04-28",
    )


@pytest.fixture
def mock_stage2():
    """Stage 2: top 15 HRP result."""
    weights = {f"SYM{i}.NS": round(100.0 / 15, 2) for i in range(15)}
    return HRPAllocationResponse(
        percentage_weights=weights,
        symbols_used=15,
        symbols_excluded=[],
        lookback_days=252,
        as_of_date="2026-04-28",
    )


@pytest.fixture
def mock_sticky_cold_start():
    """Cold-start sticky response: byte-equivalent to the legacy top-15."""
    selected = [f"SYM{i}.NS" for i in range(15)]
    return StickyTopNResponse(
        selected=selected,
        reasons={s: "top_rank" for s in selected},
        kept_count=0,
        fillers_count=15,
        evicted_from_previous={},
        previous_year_week_used=None,
        universe="halal_india_double_hrp",
        year_week="202618",
    )


@pytest.fixture
def mock_sticky_steady_state():
    """Steady-state sticky: 12 kept from prior week, 3 new fillers."""
    selected = [f"SYM{i}.NS" for i in range(15)]
    reasons = {f"SYM{i}.NS": "sticky" for i in range(12)}
    reasons.update({f"SYM{i}.NS": "top_rank" for i in range(12, 15)})
    return StickyTopNResponse(
        selected=selected,
        reasons=reasons,
        kept_count=12,
        fillers_count=3,
        evicted_from_previous={"SYM18.NS": "rank_dropped"},
        previous_year_week_used="202617",
        universe="halal_india_double_hrp",
        year_week="202618",
    )


@pytest.fixture
def mock_record_final():
    return RecordFinalWeightsResponse(
        rows_updated=15,
        universe="halal_india_double_hrp",
        year_week="202618",
    )


@pytest.fixture
def mock_summary():
    return WeeklySummaryResponse(
        summary={
            "para_1_screening_overview": "Stage 1 distributed weights across 20 stocks."
        },
        provider="openai",
        model_used="gpt-5-mini",
        tokens_used=400,
    )


@pytest.fixture
def mock_email():
    return WeeklyReportEmailResponse(
        is_success=True,
        subject="India Double HRP Portfolio Analysis (2026-04-28 -> 2026-05-02)",
        body="<html><body>India Double HRP report</body></html>",
    )


def _make_double_hrp_activities(
    *,
    universe_data,
    stage1,
    stage2,
    sticky,
    record_final,
    summary,
    email,
    hrp_calls=None,
    sticky_calls=None,
    record_final_calls=None,
    summary_calls=None,
    email_calls=None,
):
    """Build mock activities for IndiaDoubleHRPWorkflow."""

    @activity.defn(name="fetch_nifty_shariah_500_universe")
    def mock_fetch_universe() -> dict:
        return universe_data

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

    @activity.defn(name="generate_double_hrp_summary")
    def mock_generate_summary(*args, **kwargs):
        if summary_calls is not None:
            summary_calls.append({"args": args, "kwargs": kwargs})
        return summary

    @activity.defn(name="get_previous_final_allocation")
    def mock_get_previous_final_allocation(universe, current_year_week):
        # India workflow tests don't currently exercise the prior-
        # allocation block; return a cold-start payload so the email
        # activity still receives a serialisable PriorAllocation.
        from models import PreviousFinalAllocationResponse

        return PreviousFinalAllocationResponse(
            year_week=None,
            final_weights_pct={},
        )

    @activity.defn(name="send_double_hrp_email")
    def mock_send_email(*args, **kwargs):
        if email_calls is not None:
            email_calls.append({"args": args, "kwargs": kwargs})
        return email

    @activity.defn(name="generate_paper_allocation")
    def mock_paper_allocation(percentage_weights, total_nav):
        from models import AllocationDetailModel, PaperAllocationResponse

        return PaperAllocationResponse(
            details=[
                AllocationDetailModel(
                    symbol=sym,
                    weight_pct=wt,
                    price=100.0,
                    whole_shares=int((wt / 100.0 * total_nav) / 100.0),
                    trade_value=float(int((wt / 100.0 * total_nav) / 100.0) * 100),
                    stop_loss_price=94.0,
                    stop_loss_distance_pct=0.06,
                    stop_loss_reason="atr14",
                )
                for sym, wt in sorted(
                    percentage_weights.items(),
                    key=lambda kv: kv[1],
                    reverse=True,
                )
            ],
            total_nav=total_nav,
            prices_used={sym: 100.0 for sym in percentage_weights},
            total_allocated_pct=sum(percentage_weights.values()),
        )

    return [
        mock_fetch_universe,
        mock_allocate_hrp,
        mock_select_sticky,
        mock_record_final,
        mock_generate_summary,
        mock_get_previous_final_allocation,
        mock_send_email,
        mock_paper_allocation,
    ]


class TestIndiaDoubleHRPHappyPath:
    @pytest.mark.asyncio
    async def test_full_workflow_cold_start(
        self,
        mock_universe_data,
        mock_stage1,
        mock_stage2,
        mock_sticky_cold_start,
        mock_record_final,
        mock_summary,
        mock_email,
    ):
        activities = _make_double_hrp_activities(
            universe_data=mock_universe_data,
            stage1=mock_stage1,
            stage2=mock_stage2,
            sticky=mock_sticky_cold_start,
            record_final=mock_record_final,
            summary=mock_summary,
            email=mock_email,
        )

        async with worker_with_activities([IndiaDoubleHRPWorkflow], activities) as env:
            result = await env.client.execute_workflow(
                IndiaDoubleHRPWorkflow.run,
                id="test-india-double-hrp-cold-start",
                task_queue="test-queue",
            )

        assert result["universe_symbols"] == 20
        assert result["stage1_symbols_used"] == 20
        assert result["top_n"] == 15
        assert result["selected_symbols"] == mock_sticky_cold_start.selected
        assert result["stage2_symbols_used"] == 15
        assert result["kept_count"] == 0
        assert result["fillers_count"] == 15
        assert result["previous_year_week_used"] is None
        assert result["summary_provider"] == "openai"
        assert result["email"]["is_success"] is True
        assert "India Double HRP" in result["email"]["subject"]

    @pytest.mark.asyncio
    async def test_full_workflow_steady_state(
        self,
        mock_universe_data,
        mock_stage1,
        mock_stage2,
        mock_sticky_steady_state,
        mock_record_final,
        mock_summary,
        mock_email,
    ):
        activities = _make_double_hrp_activities(
            universe_data=mock_universe_data,
            stage1=mock_stage1,
            stage2=mock_stage2,
            sticky=mock_sticky_steady_state,
            record_final=mock_record_final,
            summary=mock_summary,
            email=mock_email,
        )

        async with worker_with_activities([IndiaDoubleHRPWorkflow], activities) as env:
            result = await env.client.execute_workflow(
                IndiaDoubleHRPWorkflow.run,
                id="test-india-double-hrp-steady",
                task_queue="test-queue",
            )

        assert result["kept_count"] == 12
        assert result["fillers_count"] == 3
        assert result["previous_year_week_used"] == "202617"


class TestIndiaDoubleHRPLookbacksAndSelection:
    @pytest.mark.asyncio
    async def test_stage1_uses_756_lookback_stage2_uses_252_on_sticky_selected(
        self,
        mock_universe_data,
        mock_stage1,
        mock_stage2,
        mock_sticky_cold_start,
        mock_record_final,
        mock_summary,
        mock_email,
    ):
        hrp_calls: list[dict] = []
        activities = _make_double_hrp_activities(
            universe_data=mock_universe_data,
            stage1=mock_stage1,
            stage2=mock_stage2,
            sticky=mock_sticky_cold_start,
            record_final=mock_record_final,
            summary=mock_summary,
            email=mock_email,
            hrp_calls=hrp_calls,
        )

        async with worker_with_activities([IndiaDoubleHRPWorkflow], activities) as env:
            await env.client.execute_workflow(
                IndiaDoubleHRPWorkflow.run,
                id="test-india-double-hrp-lookbacks",
                task_queue="test-queue",
            )

        assert len(hrp_calls) == 2

        assert hrp_calls[0]["lookback_days"] == 756
        expected_universe = [s["symbol"] for s in mock_universe_data["stocks"]]
        assert hrp_calls[0]["symbols"] == expected_universe

        # Stage 2 must consume exactly sticky.selected -- not the raw top-15
        # by Stage 1 weight. This is the math-correctness invariant: the
        # final allocation has to be on the same set the workflow says it
        # picked; otherwise the email/LLM lie about what got allocated.
        assert hrp_calls[1]["lookback_days"] == 252
        assert hrp_calls[1]["symbols"] == mock_sticky_cold_start.selected


class TestIndiaDoubleHRPPartitionIsolation:
    @pytest.mark.asyncio
    async def test_sticky_uses_halal_india_double_hrp_partition_and_threshold(
        self,
        mock_universe_data,
        mock_stage1,
        mock_stage2,
        mock_sticky_cold_start,
        mock_record_final,
        mock_summary,
        mock_email,
    ):
        sticky_calls: list[dict] = []
        record_final_calls: list[dict] = []

        activities = _make_double_hrp_activities(
            universe_data=mock_universe_data,
            stage1=mock_stage1,
            stage2=mock_stage2,
            sticky=mock_sticky_cold_start,
            record_final=mock_record_final,
            summary=mock_summary,
            email=mock_email,
            sticky_calls=sticky_calls,
            record_final_calls=record_final_calls,
        )

        async with worker_with_activities([IndiaDoubleHRPWorkflow], activities) as env:
            await env.client.execute_workflow(
                IndiaDoubleHRPWorkflow.run,
                id="test-india-double-hrp-partition",
                task_queue="test-queue",
            )

        # Pin the partition string. Reusing "halal_india_alpha" or
        # "halal_new" would corrupt those strategies' sticky carry-sets;
        # see brain_api.core.strategy_partitions.
        assert len(sticky_calls) == 1
        assert sticky_calls[0]["universe"] == "halal_india_double_hrp"
        assert sticky_calls[0]["top_n"] == 15
        assert sticky_calls[0]["threshold"] == 1.0
        # year_week must be ISO 'YYYYWW' (6 chars) -- matches sibling
        # workflows so prior-week reads align across reruns.
        assert len(sticky_calls[0]["year_week"]) == 6

        # record_final_weights MUST share partition + year_week with the
        # sticky call. If either drifts, next week's sticky read finds an
        # empty final_set and degrades to a cold start (silent drift).
        assert len(record_final_calls) == 1
        assert record_final_calls[0]["universe"] == "halal_india_double_hrp"
        assert record_final_calls[0]["n_weights"] == 15
        assert record_final_calls[0]["year_week"] == sticky_calls[0]["year_week"]


class TestIndiaDoubleHRPRunIdContract:
    @pytest.mark.asyncio
    async def test_run_id_is_default_paper_form(
        self,
        mock_universe_data,
        mock_stage1,
        mock_stage2,
        mock_sticky_cold_start,
        mock_record_final,
        mock_summary,
        mock_email,
    ):
        sticky_calls: list[dict] = []

        activities = _make_double_hrp_activities(
            universe_data=mock_universe_data,
            stage1=mock_stage1,
            stage2=mock_stage2,
            sticky=mock_sticky_cold_start,
            record_final=mock_record_final,
            summary=mock_summary,
            email=mock_email,
            sticky_calls=sticky_calls,
        )

        async with worker_with_activities([IndiaDoubleHRPWorkflow], activities) as env:
            await env.client.execute_workflow(
                IndiaDoubleHRPWorkflow.run,
                id="test-india-double-hrp-run-id",
                task_queue="test-queue",
            )

        # India is paper-only with no broker and uses the default run_id
        # form ``paper:YYYY-MM-DD`` -- the per-universe variant
        # (``paper:<universe>:YYYY-MM-DD``) is reserved per AGENTS.md
        # rule for strategies that share an Alpaca account.
        assert len(sticky_calls) == 1
        run_id = sticky_calls[0]["run_id"]
        assert run_id.startswith("paper:")
        assert run_id.count(":") == 1
        # YYYY-MM-DD after the colon
        date_part = run_id.split(":", 1)[1]
        datetime.strptime(date_part, "%Y-%m-%d")

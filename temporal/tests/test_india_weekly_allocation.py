"""Tests for the India Alpha-HRP Temporal workflow.

Covers cold-start (no prior week) and stable-week scenarios mirroring
``tests/test_us_alpha_hrp_happy.py``. Failure propagation is also
exercised at the universe-fetch boundary (no Alpaca account, so the
US sell-wait-buy / skip-path branches are not present).
"""

from __future__ import annotations

import pytest
from temporalio import activity
from temporalio.client import WorkflowFailureError

from models import (
    HRPAllocationResponse,
    PatchTSTBatchScores,
    RankBandTopNResponse,
    RecordFinalWeightsResponse,
    WeeklyReportEmailResponse,
    WeeklySummaryResponse,
)
from tests.harness import (
    make_india_alpha_hrp_activities,
    worker_with_activities,
)
from workflows.india_weekly_allocation import IndiaWeeklyAllocationWorkflow


@pytest.fixture
def india_universe_data():
    """Simulated nifty_shariah_500 universe response (~210 in production)."""
    return {
        "stocks": [{"symbol": f"NSE{i:03d}.NS"} for i in range(20)],
        "total_stocks": 20,
        "source": "nifty_shariah_500",
    }


@pytest.fixture
def india_patchtst_scores():
    """20 valid India PatchTST scores spread evenly so ranks are unambiguous."""
    return PatchTSTBatchScores(
        scores={f"NSE{i:03d}.NS": float(20 - i) for i in range(20)},
        model_version="v2026-04-26-india",
        as_of_date="2026-04-28",
        target_week_start="2026-04-28",
        target_week_end="2026-05-02",
        requested_count=20,
        predicted_count=20,
        excluded_symbols=[],
    )


@pytest.fixture
def india_stage2_alloc():
    weights = {f"NSE{i:03d}.NS": round(100.0 / 15, 2) for i in range(15)}
    return HRPAllocationResponse(
        percentage_weights=weights,
        symbols_used=15,
        symbols_excluded=[],
        lookback_days=252,
        as_of_date="2026-04-28",
    )


@pytest.fixture
def india_sticky_cold_start():
    selected = [f"NSE{i:03d}.NS" for i in range(15)]
    return RankBandTopNResponse(
        selected=selected,
        reasons={s: "top_rank" for s in selected},
        kept_count=0,
        fillers_count=15,
        evicted_from_previous={},
        previous_year_week_used=None,
        universe="halal_india_alpha",
        year_week="202618",
        top_n=15,
        hold_threshold=30,
    )


@pytest.fixture
def india_sticky_stable():
    selected = [f"NSE{i:03d}.NS" for i in range(15)]
    return RankBandTopNResponse(
        selected=selected,
        reasons={s: "sticky" for s in selected},
        kept_count=15,
        fillers_count=0,
        evicted_from_previous={},
        previous_year_week_used="202617",
        universe="halal_india_alpha",
        year_week="202618",
        top_n=15,
        hold_threshold=30,
    )


@pytest.fixture
def india_record_final_resp():
    return RecordFinalWeightsResponse(
        rows_updated=15, universe="halal_india_alpha", year_week="202618"
    )


@pytest.fixture
def india_summary_resp():
    return WeeklySummaryResponse(
        summary={"para_1_market_outlook": "Top NSE names look strong."},
        provider="openai",
        model_used="gpt-4o-mini",
        tokens_used=350,
    )


@pytest.fixture
def india_email_resp():
    return WeeklyReportEmailResponse(
        is_success=True,
        subject="India Alpha-HRP Portfolio Analysis (2026-04-28 -> 2026-05-02)",
        body="<html><body>India Alpha-HRP report</body></html>",
    )


class TestIndiaAlphaHRPHappyPath:
    @pytest.mark.asyncio
    async def test_full_workflow_cold_start(
        self,
        india_universe_data,
        india_patchtst_scores,
        india_stage2_alloc,
        india_sticky_cold_start,
        india_record_final_resp,
        india_summary_resp,
        india_email_resp,
    ):
        score_calls: list[dict] = []
        select_calls: list[dict] = []
        hrp_calls: list[dict] = []
        record_final_calls: list[dict] = []

        activities = make_india_alpha_hrp_activities(
            universe_data=india_universe_data,
            scores=india_patchtst_scores,
            stage2=india_stage2_alloc,
            sticky=india_sticky_cold_start,
            record_final=india_record_final_resp,
            summary=india_summary_resp,
            email=india_email_resp,
            score_calls=score_calls,
            select_calls=select_calls,
            hrp_calls=hrp_calls,
            record_final_calls=record_final_calls,
        )

        async with worker_with_activities(
            [IndiaWeeklyAllocationWorkflow], activities
        ) as env:
            result = await env.client.execute_workflow(
                IndiaWeeklyAllocationWorkflow.run,
                id="test-india-alpha-hrp-cold-start",
                task_queue="test-queue",
            )

        # Phase-by-phase assertions on the workflow output.
        assert result["universe_symbols"] == 20
        assert result["stage1_predicted_count"] == 20
        assert result["model_version"] == "v2026-04-26-india"
        assert result["top_n"] == 15
        assert result["hold_threshold"] == 30
        assert result["kept_count"] == 0
        assert result["fillers_count"] == 15
        assert result["previous_year_week_used"] is None
        assert result["stage2_symbols_used"] == 15
        assert len(result["selected_symbols"]) == 15
        assert result["email"]["is_success"] is True

        # Phase 1: PatchTST scoring activity called with full universe.
        assert len(score_calls) == 1
        assert score_calls[0]["symbols"] == [
            s["symbol"] for s in india_universe_data["stocks"]
        ]
        assert score_calls[0]["min_predictions"] == 15

        # Phase 1.5: rank-band selection on the halal_india_alpha
        # partition (NOT halal_new_alpha -- distinct by mathematical
        # requirement to keep sticky rows isolated per market).
        assert len(select_calls) == 1
        assert select_calls[0]["universe"] == "halal_india_alpha"
        assert select_calls[0]["top_n"] == 15
        assert select_calls[0]["hold_threshold"] == 30
        assert select_calls[0]["scores_count"] == 20

        # Phase 2: HRP runs ONLY on the selected set with 252d lookback.
        assert len(hrp_calls) == 1
        assert hrp_calls[0]["lookback_days"] == 252
        assert hrp_calls[0]["symbols"] == india_sticky_cold_start.selected

        # Phase 2.5: final weights recorded under the correct partition.
        assert len(record_final_calls) == 1
        assert record_final_calls[0]["universe"] == "halal_india_alpha"
        assert record_final_calls[0]["year_week"] == select_calls[0]["year_week"]
        assert record_final_calls[0]["n_weights"] == 15

    @pytest.mark.asyncio
    async def test_stable_week_all_kept(
        self,
        india_universe_data,
        india_patchtst_scores,
        india_stage2_alloc,
        india_sticky_stable,
        india_record_final_resp,
        india_summary_resp,
        india_email_resp,
    ):
        activities = make_india_alpha_hrp_activities(
            universe_data=india_universe_data,
            scores=india_patchtst_scores,
            stage2=india_stage2_alloc,
            sticky=india_sticky_stable,
            record_final=india_record_final_resp,
            summary=india_summary_resp,
            email=india_email_resp,
        )

        async with worker_with_activities(
            [IndiaWeeklyAllocationWorkflow], activities
        ) as env:
            result = await env.client.execute_workflow(
                IndiaWeeklyAllocationWorkflow.run,
                id="test-india-alpha-hrp-stable",
                task_queue="test-queue",
            )

        # All 15 selected are kept from prior week.
        assert result["kept_count"] == 15
        assert result["fillers_count"] == 0
        assert result["previous_year_week_used"] == "202617"
        assert result["stage2_symbols_used"] == 15


class TestIndiaWorkflowFailurePropagation:
    @pytest.mark.asyncio
    async def test_universe_failure_stops_workflow(
        self,
        india_patchtst_scores,
        india_stage2_alloc,
        india_sticky_cold_start,
        india_record_final_resp,
        india_summary_resp,
        india_email_resp,
    ):
        """If fetch_nifty_shariah_500_universe raises, the workflow fails.

        We override only the universe activity inside a hand-rolled
        harness so the mocked downstream activities are never reached.
        Confirms the phase-0 boundary stops the rest of the pipeline.
        """

        @activity.defn(name="fetch_nifty_shariah_500_universe")
        def mock_fetch_universe() -> dict:
            raise RuntimeError("NSE API down")

        @activity.defn(name="score_halal_india_with_patchtst")
        def mock_score(symbols, as_of_date, min_predictions=15):
            return india_patchtst_scores

        @activity.defn(name="select_rank_band_top_n")
        def mock_select(*args, **kwargs):
            return india_sticky_cold_start

        @activity.defn(name="allocate_hrp")
        def mock_allocate(*args, **kwargs):
            return india_stage2_alloc

        @activity.defn(name="record_final_weights")
        def mock_record(*args, **kwargs):
            return india_record_final_resp

        @activity.defn(name="generate_india_alpha_hrp_summary")
        def mock_summary(*args, **kwargs):
            return india_summary_resp

        @activity.defn(name="send_india_alpha_hrp_email")
        def mock_email(*args, **kwargs):
            return india_email_resp

        async with worker_with_activities(
            [IndiaWeeklyAllocationWorkflow],
            [
                mock_fetch_universe,
                mock_score,
                mock_select,
                mock_allocate,
                mock_record,
                mock_summary,
                mock_email,
            ],
        ) as env:
            with pytest.raises(WorkflowFailureError):
                await env.client.execute_workflow(
                    IndiaWeeklyAllocationWorkflow.run,
                    id="test-india-alpha-hrp-fail",
                    task_queue="test-queue",
                )

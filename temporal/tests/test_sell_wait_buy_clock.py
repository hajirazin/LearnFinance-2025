"""Behavioural tests for the clock-driven sell-wait-buy cadence.

The shared :func:`workflows._order_execution.sell_wait_buy` helper
fetches the Alpaca market clock once after submitting sells and -- if
the market is closed -- sleeps until exactly ``next_open`` before the
1-min status polling loop kicks in. These tests exercise that
behaviour through ``USWeeklyAllocationWorkflow`` (the cheapest
sell-wait-buy carrier) by capturing each activity's
``activity.info().current_attempt_scheduled_time`` -- which, in the
time-skipping ``WorkflowEnvironment``, advances when the workflow
sleeps -- and asserting the relative gaps.

Math correctness invariant under test:
- pre-open gap = ``next_open - workflow.now()`` exactly (no lead-time
  fudge).
- subsequent polls are spaced by ``POLL_INTERVAL = 1 min``.
- the clock fetch is gated on ``len(sell_order_ids) > 0``: a buy-only
  cycle must NOT call ``get_alpaca_clock``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from temporalio import activity

from models import MarketClockResponse
from tests.harness import make_sac_only_activities, worker_with_activities
from workflows.us_weekly_allocation import USWeeklyAllocationWorkflow


def _activity_now() -> datetime:
    """Return the workflow-logical scheduled time of the current activity.

    In a time-skipping ``WorkflowEnvironment``,
    ``activity.info().current_attempt_scheduled_time`` matches the
    workflow's logical ``now()`` when the activity was scheduled, so
    we can use it to observe how long the workflow slept between two
    activity invocations without probing internal workflow state.
    """
    return activity.info().current_attempt_scheduled_time


class TestPreOpenSleepsUntilNextOpen:
    """If the market is closed, the helper sleeps until exactly ``next_open``."""

    @pytest.mark.asyncio
    async def test_pre_open_first_poll_lands_at_next_open(
        self,
        active_symbols,
        sac_portfolio_no_open,
        lstm_resp,
        patchtst_resp,
        news_resp,
        fundamentals_resp,
        sac_alloc,
        sell_and_buy_orders,
        sac_submit_resp,
        sac_summary_resp,
        sac_email_resp,
    ):
        captured: dict[str, datetime | list[datetime]] = {"polls": []}

        def get_alpaca_clock_fn() -> MarketClockResponse:
            t_clock = _activity_now()
            captured["clock_t"] = t_clock
            next_open = t_clock + timedelta(minutes=30)
            captured["expected_next_open"] = next_open
            return MarketClockResponse(
                timestamp=t_clock.isoformat(),
                is_open=False,
                next_open=next_open.isoformat(),
                next_close=(t_clock + timedelta(hours=10)).isoformat(),
            )

        def check_fn(account, client_order_ids):
            captured["polls"].append(_activity_now())
            # Terminal on the first poll so the loop exits without
            # additional 1-min waits.
            return [
                {"client_order_id": cid, "status": "filled"} for cid in client_order_ids
            ]

        activities = make_sac_only_activities(
            active_symbols=active_symbols,
            sac_portfolio=sac_portfolio_no_open,
            fundamentals_resp=fundamentals_resp,
            news_resp=news_resp,
            lstm_resp=lstm_resp,
            patchtst_resp=patchtst_resp,
            sac_alloc=sac_alloc,
            sac_orders=sell_and_buy_orders,
            sac_submit_resp=sac_submit_resp,
            summary_resp=sac_summary_resp,
            email_resp=sac_email_resp,
            check_order_statuses_fn=check_fn,
            get_alpaca_clock_fn=get_alpaca_clock_fn,
        )

        async with worker_with_activities(
            [USWeeklyAllocationWorkflow], activities
        ) as env:
            await env.client.execute_workflow(
                USWeeklyAllocationWorkflow.run,
                id="test-clock-pre-open-sleep",
                task_queue="test-queue",
            )

        assert captured["polls"], "Workflow must have polled at least once"
        first_poll = captured["polls"][0]
        next_open = captured["expected_next_open"]
        # The first poll must land at (or microseconds after) next_open.
        # Allow a small positive tolerance for activity scheduling
        # latency in the time-skipping env, but never EARLY -- that
        # would mean we ignored the clock.
        delta = first_poll - next_open
        assert delta >= timedelta(0), (
            f"First poll fired BEFORE next_open: delta={delta}"
        )
        assert delta < timedelta(seconds=10), (
            f"First poll fired too late after next_open: delta={delta}"
        )


class TestMarketOpenStartsImmediately:
    """If the market is currently open, the helper polls immediately."""

    @pytest.mark.asyncio
    async def test_open_market_polls_without_big_sleep(
        self,
        active_symbols,
        sac_portfolio_no_open,
        lstm_resp,
        patchtst_resp,
        news_resp,
        fundamentals_resp,
        sac_alloc,
        sell_and_buy_orders,
        sac_submit_resp,
        sac_summary_resp,
        sac_email_resp,
    ):
        captured: dict[str, datetime | list[datetime]] = {"polls": []}

        def get_alpaca_clock_fn() -> MarketClockResponse:
            t_clock = _activity_now()
            captured["clock_t"] = t_clock
            return MarketClockResponse(
                timestamp=t_clock.isoformat(),
                is_open=True,
                # next_open in the past so any logic that reaches for
                # it without first checking is_open would underflow.
                next_open=(t_clock - timedelta(hours=1)).isoformat(),
                next_close=(t_clock + timedelta(hours=6)).isoformat(),
            )

        def check_fn(account, client_order_ids):
            captured["polls"].append(_activity_now())
            return [
                {"client_order_id": cid, "status": "filled"} for cid in client_order_ids
            ]

        activities = make_sac_only_activities(
            active_symbols=active_symbols,
            sac_portfolio=sac_portfolio_no_open,
            fundamentals_resp=fundamentals_resp,
            news_resp=news_resp,
            lstm_resp=lstm_resp,
            patchtst_resp=patchtst_resp,
            sac_alloc=sac_alloc,
            sac_orders=sell_and_buy_orders,
            sac_submit_resp=sac_submit_resp,
            summary_resp=sac_summary_resp,
            email_resp=sac_email_resp,
            check_order_statuses_fn=check_fn,
            get_alpaca_clock_fn=get_alpaca_clock_fn,
        )

        async with worker_with_activities(
            [USWeeklyAllocationWorkflow], activities
        ) as env:
            await env.client.execute_workflow(
                USWeeklyAllocationWorkflow.run,
                id="test-clock-open-immediate",
                task_queue="test-queue",
            )

        assert captured["polls"], "Workflow must have polled"
        gap = captured["polls"][0] - captured["clock_t"]
        assert gap < timedelta(seconds=5), (
            f"Open-market workflow slept before first poll: gap={gap}"
        )


class TestNextOpenInPastNoWait:
    """``next_open`` in the past must not produce a negative sleep."""

    @pytest.mark.asyncio
    async def test_next_open_in_past_polls_immediately(
        self,
        active_symbols,
        sac_portfolio_no_open,
        lstm_resp,
        patchtst_resp,
        news_resp,
        fundamentals_resp,
        sac_alloc,
        sell_and_buy_orders,
        sac_submit_resp,
        sac_summary_resp,
        sac_email_resp,
    ):
        captured: dict[str, datetime | list[datetime]] = {"polls": []}

        def get_alpaca_clock_fn() -> MarketClockResponse:
            t_clock = _activity_now()
            captured["clock_t"] = t_clock
            # Pathological payload: market is reported closed but
            # next_open is already in the past. The helper must NOT
            # try to sleep a negative timedelta.
            return MarketClockResponse(
                timestamp=t_clock.isoformat(),
                is_open=False,
                next_open=(t_clock - timedelta(hours=1)).isoformat(),
                next_close=(t_clock + timedelta(hours=6)).isoformat(),
            )

        def check_fn(account, client_order_ids):
            captured["polls"].append(_activity_now())
            return [
                {"client_order_id": cid, "status": "filled"} for cid in client_order_ids
            ]

        activities = make_sac_only_activities(
            active_symbols=active_symbols,
            sac_portfolio=sac_portfolio_no_open,
            fundamentals_resp=fundamentals_resp,
            news_resp=news_resp,
            lstm_resp=lstm_resp,
            patchtst_resp=patchtst_resp,
            sac_alloc=sac_alloc,
            sac_orders=sell_and_buy_orders,
            sac_submit_resp=sac_submit_resp,
            summary_resp=sac_summary_resp,
            email_resp=sac_email_resp,
            check_order_statuses_fn=check_fn,
            get_alpaca_clock_fn=get_alpaca_clock_fn,
        )

        async with worker_with_activities(
            [USWeeklyAllocationWorkflow], activities
        ) as env:
            await env.client.execute_workflow(
                USWeeklyAllocationWorkflow.run,
                id="test-clock-past-next-open",
                task_queue="test-queue",
            )

        assert captured["polls"], "Workflow must have polled"
        gap = captured["polls"][0] - captured["clock_t"]
        assert gap < timedelta(seconds=5), (
            f"next_open-in-past should not induce a sleep: gap={gap}"
        )


class TestOneMinuteCadenceBetweenPolls:
    """Once polling, the helper waits ``POLL_INTERVAL = 1 min`` between checks."""

    @pytest.mark.asyncio
    async def test_pending_then_filled_one_minute_apart(
        self,
        active_symbols,
        sac_portfolio_no_open,
        lstm_resp,
        patchtst_resp,
        news_resp,
        fundamentals_resp,
        sac_alloc,
        sell_and_buy_orders,
        sac_submit_resp,
        sac_summary_resp,
        sac_email_resp,
    ):
        captured: dict[str, list[datetime]] = {"polls": []}

        def get_alpaca_clock_fn() -> MarketClockResponse:
            t_clock = _activity_now()
            return MarketClockResponse(
                timestamp=t_clock.isoformat(),
                is_open=True,
                next_open=(t_clock + timedelta(hours=23)).isoformat(),
                next_close=(t_clock + timedelta(hours=6)).isoformat(),
            )

        def check_fn(account, client_order_ids):
            captured["polls"].append(_activity_now())
            # First two polls return pending; the third returns filled
            # so the loop exits after exactly three iterations and we
            # can measure two gaps.
            if len(captured["polls"]) <= 2:
                return [
                    {"client_order_id": cid, "status": "pending_new"}
                    for cid in client_order_ids
                ]
            return [
                {"client_order_id": cid, "status": "filled"} for cid in client_order_ids
            ]

        activities = make_sac_only_activities(
            active_symbols=active_symbols,
            sac_portfolio=sac_portfolio_no_open,
            fundamentals_resp=fundamentals_resp,
            news_resp=news_resp,
            lstm_resp=lstm_resp,
            patchtst_resp=patchtst_resp,
            sac_alloc=sac_alloc,
            sac_orders=sell_and_buy_orders,
            sac_submit_resp=sac_submit_resp,
            summary_resp=sac_summary_resp,
            email_resp=sac_email_resp,
            check_order_statuses_fn=check_fn,
            get_alpaca_clock_fn=get_alpaca_clock_fn,
        )

        async with worker_with_activities(
            [USWeeklyAllocationWorkflow], activities
        ) as env:
            await env.client.execute_workflow(
                USWeeklyAllocationWorkflow.run,
                id="test-clock-one-minute-cadence",
                task_queue="test-queue",
            )

        polls = captured["polls"]
        assert len(polls) >= 3, f"Expected at least 3 polls, got {len(polls)}"

        # Each successive gap must be at least POLL_INTERVAL (1 min)
        # and tightly bounded above so we know we're not back on the
        # 15-min cadence.
        for i in range(1, len(polls)):
            gap = polls[i] - polls[i - 1]
            assert timedelta(seconds=55) <= gap <= timedelta(seconds=70), (
                f"Poll gap {i} out of expected 1-min band: gap={gap}"
            )


class TestNoSellsSkipsClockFetch:
    """A buy-only cycle must not invoke ``get_alpaca_clock`` at all."""

    @pytest.mark.asyncio
    async def test_buy_only_does_not_fetch_clock(
        self,
        active_symbols,
        sac_portfolio_no_open,
        lstm_resp,
        patchtst_resp,
        news_resp,
        fundamentals_resp,
        sac_alloc,
        buy_only_orders,
        sac_submit_resp,
        sac_summary_resp,
        sac_email_resp,
    ):
        clock_calls: list[None] = []

        # Use a sentinel response so a stray call is loud, but rely on
        # ``get_alpaca_clock_calls`` for the assertion.
        def get_alpaca_clock_fn() -> MarketClockResponse:
            return MarketClockResponse(
                timestamp=datetime.now(tz=UTC).isoformat(),
                is_open=True,
                next_open=datetime.now(tz=UTC).isoformat(),
                next_close=datetime.now(tz=UTC).isoformat(),
            )

        activities = make_sac_only_activities(
            active_symbols=active_symbols,
            sac_portfolio=sac_portfolio_no_open,
            fundamentals_resp=fundamentals_resp,
            news_resp=news_resp,
            lstm_resp=lstm_resp,
            patchtst_resp=patchtst_resp,
            sac_alloc=sac_alloc,
            sac_orders=buy_only_orders,
            sac_submit_resp=sac_submit_resp,
            summary_resp=sac_summary_resp,
            email_resp=sac_email_resp,
            get_alpaca_clock_fn=get_alpaca_clock_fn,
            get_alpaca_clock_calls=clock_calls,
        )

        async with worker_with_activities(
            [USWeeklyAllocationWorkflow], activities
        ) as env:
            await env.client.execute_workflow(
                USWeeklyAllocationWorkflow.run,
                id="test-clock-no-sells-skip",
                task_queue="test-queue",
            )

        assert clock_calls == [], (
            "Buy-only cycle must not fetch the Alpaca clock; "
            f"got {len(clock_calls)} call(s)"
        )

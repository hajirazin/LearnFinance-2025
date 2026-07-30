"""Deterministic tests for the weekly SAC XNYS trade clock."""

from datetime import date

import pandas as pd
import pytest

from brain_api.core.sac.trade_clock import (
    build_sac_weekly_trade_clock,
    experience_open_transition,
)


def test_trade_clock_uses_first_xnys_session_when_monday_is_a_holiday():
    clock = build_sac_weekly_trade_clock(
        date(2026, 1, 12),
        date(2026, 1, 27),
    )

    assert [session.date() for session in clock.rebalance_sessions] == [
        date(2026, 1, 12),
        date(2026, 1, 20),
        date(2026, 1, 26),
    ]
    assert [cutoff.date() for cutoff in clock.actor_cutoffs] == [
        date(2026, 1, 9),
        date(2026, 1, 16),
        date(2026, 1, 23),
    ]


def test_trade_clock_excludes_partial_iso_week_at_window_start():
    clock = build_sac_weekly_trade_clock(
        date(2016, 1, 1),
        date(2016, 1, 15),
    )

    assert [session.date() for session in clock.rebalance_sessions] == [
        date(2016, 1, 4),
        date(2016, 1, 11),
    ]


def test_experience_transition_uses_start_open_and_next_week_first_open():
    frame = pd.DataFrame(
        {"open": [100.0, 999.0, 110.0]},
        index=pd.to_datetime(["2026-01-20", "2026-01-23", "2026-01-26"]),
    )

    trade_price, simple_return = experience_open_transition(
        frame,
        date(2026, 1, 19),
        symbol="AAPL",
    )

    assert trade_price == 100.0
    assert simple_return == pytest.approx(0.10)


def test_experience_transition_fails_when_a_required_open_is_missing():
    frame = pd.DataFrame(
        {"open": [100.0]},
        index=pd.to_datetime(["2026-01-20"]),
    )

    with pytest.raises(ValueError, match="2026-01-26"):
        experience_open_transition(frame, date(2026, 1, 19), symbol="AAPL")

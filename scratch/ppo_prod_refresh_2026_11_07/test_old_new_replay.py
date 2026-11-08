"""Causality and financial-accounting checks for the pinned replay."""

from datetime import date

import pandas as pd
import pytest

from old_new_replay import CAPITAL, SESSIONS, evaluate_week, history_before_decision


def test_decision_session_and_future_prices_cannot_enter_history():
    frame = pd.DataFrame(
        {"close": [100.0, 200.0, 300.0]},
        index=pd.to_datetime(["2026-10-30", "2026-11-02", "2026-11-03"]),
    )
    before = history_before_decision(frame, date(2026, 11, 2))
    frame.loc["2026-11-02":, "close"] = 1_000_000
    pd.testing.assert_frame_equal(
        before, history_before_decision(frame, date(2026, 11, 2))
    )
    assert list(before["close"]) == [100.0]


def test_weekly_loss_respects_cash_and_deducts_entry_fee_once():
    frame = pd.DataFrame({"open": 100.0, "close": 90.0}, index=SESSIONS)
    result = evaluate_week({"A": 0.5, "CASH": 0.5}, {"A": frame})
    assert result["gross_return"] == pytest.approx(-0.05)
    assert result["entry_cost_usd"] > 0
    assert result["ending_value"] == pytest.approx(
        CAPITAL - 500 - result["entry_cost_usd"]
    )
    assert result["wealth"][0] == CAPITAL

"""Calendar and no-future-price invariants for the full-year replay."""

from datetime import date

import pandas as pd

from simulate import past_frames, schedule


def test_holiday_execution_and_friday_valuation_follow_actual_sessions():
    rows = schedule()
    martin_luther = next(r for r in rows if r["decision"].date() == date(2026, 1, 19))
    assert martin_luther["execution"] == pd.Timestamp("2026-01-20")
    easter = next(r for r in rows if r["decision"].date() == date(2026, 3, 30))
    assert easter["mark"] == pd.Timestamp("2026-04-02")
    assert len(rows) == 44 and rows[-1]["mark"] == pd.Timestamp("2026-11-06")


def test_all_decision_session_prices_are_excluded_even_if_they_exist():
    frame = pd.DataFrame(
        {"close": [10.0, 100.0, 1000.0]},
        index=pd.to_datetime(["2026-10-30", "2026-11-02", "2026-11-06"]),
    )
    filtered = past_frames({"A": frame}, date(2026, 11, 2))["A"]
    assert list(filtered["close"]) == [10.0]
    assert frame.iloc[-1, 0] == 1000

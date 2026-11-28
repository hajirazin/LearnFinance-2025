"""Check compounding and date attribution with disposable synthetic returns."""

from datetime import date

import pytest
from chart_data import wealth_curve


def test_returns_are_booked_at_the_following_open_not_the_decision_date():
    dates, wealth = wealth_curve(
        ["2024-11-25", "2024-12-02"], "2024-12-09", [0.1, -0.1]
    )
    assert dates == (date(2024, 11, 25), date(2024, 12, 2), date(2024, 12, 9))
    assert wealth == pytest.approx((10_000, 11_000, 9_900))


def test_mismatched_weekly_returns_are_rejected():
    with pytest.raises(ValueError, match="exactly one return"):
        wealth_curve(["2024-11-25", "2024-12-02"], "2024-12-09", [0.1])


def test_terminal_open_cannot_overlap_the_last_decision():
    with pytest.raises(ValueError, match="strictly ordered"):
        wealth_curve(["2024-11-25", "2024-12-02"], "2024-12-02", [0.1, -0.1])

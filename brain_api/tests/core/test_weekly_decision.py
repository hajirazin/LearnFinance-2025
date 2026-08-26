from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from brain_api.core.weekly_decision import (
    MondayCutoffError,
    canonical_monday_windows_contained_in,
    monday_decision_cutoff,
    previous_monday_decision_cutoff,
    require_monday_decision_cutoff,
)

NY = ZoneInfo("America/New_York")


def test_monday_cutoff_dst() -> None:
    summer = monday_decision_cutoff(date(2026, 8, 24))
    winter = monday_decision_cutoff(date(2026, 1, 5))
    assert summer.isoformat() == "2026-08-24T09:00:00-04:00"
    assert winter.isoformat() == "2026-01-05T09:00:00-05:00"
    prev = previous_monday_decision_cutoff(date(2026, 8, 24))
    assert prev.isoformat() == "2026-08-17T09:00:00-04:00"


def test_require_cutoff_rejects_wrong_hour() -> None:
    with pytest.raises(MondayCutoffError):
        require_monday_decision_cutoff(datetime(2026, 8, 24, 10, 0, tzinfo=NY))
    cutoff = require_monday_decision_cutoff(datetime(2026, 8, 24, 9, 0, tzinfo=NY))
    assert cutoff.hour == 9


def test_tiling_omits_partial_edges() -> None:
    start = datetime(2026, 8, 19, 0, 0, tzinfo=NY)
    end = datetime(2026, 8, 24, 13, 0, tzinfo=NY)
    windows = canonical_monday_windows_contained_in(start, end)
    assert windows == []
    start = datetime(2026, 8, 17, 9, 0, tzinfo=NY)
    end = datetime(2026, 8, 24, 9, 0, tzinfo=NY)
    windows = canonical_monday_windows_contained_in(start, end)
    assert len(windows) == 1
    assert windows[0][1].isoformat() == "2026-08-24T09:00:00-04:00"

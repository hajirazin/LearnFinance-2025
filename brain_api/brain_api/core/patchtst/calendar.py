"""Exchange-session evidence for weekly PatchTST training and serving."""

from dataclasses import dataclass
from datetime import date, timedelta

import exchange_calendars as xcals
import pandas as pd

from brain_api.core.inference_utils import WeekBoundaries
from brain_api.storage.nse_calendar import NseCashHolidays, load_nse_cash_holidays

FORECAST_SESSIONS = 5
PRICE_WARMUP_SESSIONS = 252
PRICE_WARMUP_CALENDAR_DAYS = 795


@dataclass(frozen=True)
class PatchTSTCalendar:
    closes: pd.Series

    @property
    def sessions(self) -> pd.DatetimeIndex:
        return self.closes.index

    def window(self, anchor: date, days: int) -> tuple[int, int]:
        left = self.sessions.searchsorted(
            pd.Timestamp(anchor - timedelta(days=days)), "right"
        )
        right = self.sessions.searchsorted(pd.Timestamp(anchor), "right")
        return left, right

    def anchor(self, cutoff_friday: date) -> date:
        end = pd.Timestamp(cutoff_friday + timedelta(days=3))
        index = self.sessions.searchsorted(end, "left") - 1
        if index < 0:
            raise ValueError(
                f"No completed session for PatchTST cutoff {cutoff_friday}"
            )
        return self.sessions[index].date()

    def forecast_end(self, anchor: date) -> date:
        index = self.sessions.get_loc(pd.Timestamp(anchor)) + FORECAST_SESSIONS
        if index >= len(self.sessions):
            raise ValueError(f"Calendar lacks five sessions after {anchor}")
        return self.sessions[index].date()

    def week(self, cutoff: date) -> WeekBoundaries:
        monday = cutoff + timedelta(days=3)
        friday = monday + timedelta(days=4)
        left = self.sessions.searchsorted(pd.Timestamp(monday), "left")
        right = self.sessions.searchsorted(
            pd.Timestamp(friday + timedelta(days=2)), "right"
        )
        sessions = self.sessions[left:right]
        if len(sessions) == 0:
            raise ValueError(f"No scheduled exchange sessions in the week of {monday}")
        return WeekBoundaries(sessions[0].date(), sessions[-1].date(), monday, friday)


def load_patchtst_calendar(market: str, start: date, end: date) -> PatchTSTCalendar:
    if market == "us":
        calendar = xcals.get_calendar(
            "XNYS",
            start=str(start - timedelta(days=7)),
            end=str(end + timedelta(days=7)),
        )
        sessions = calendar.sessions_in_range(str(start), str(end))
        closes = calendar.schedule.loc[sessions, "close"].copy()
        closes.index = closes.index.tz_localize(None)
        return PatchTSTCalendar(closes)
    if market != "india":
        raise ValueError(f"Unknown PatchTST market {market!r}")
    return _india_calendar(
        start, end, load_nse_cash_holidays(end, refresh_missing_years=True)
    )


def _india_calendar(
    start: date, end: date, coverage: NseCashHolidays
) -> PatchTSTCalendar:
    if not coverage.start <= start <= end <= coverage.end:
        raise ValueError(f"NSE cash holidays do not cover {start} through {end}")
    weekdays = pd.bdate_range(start, end)
    sessions = weekdays.difference(pd.DatetimeIndex(sorted(coverage.holidays)))
    # India uses daily price bars. The calendar tracks regular session dates only.
    closes = (sessions + pd.Timedelta(hours=15, minutes=30)).tz_localize("Asia/Kolkata")
    return PatchTSTCalendar(pd.Series(closes.tz_convert("UTC"), index=sessions))


def training_calendar(market: str, start: date, end: date) -> PatchTSTCalendar:
    # Covers the calendar context and 252 preceding sessions with room for holidays.
    return load_patchtst_calendar(
        market, start - timedelta(days=PRICE_WARMUP_CALENDAR_DAYS), end
    )


def inference_calendar(market: str, cutoff: date) -> PatchTSTCalendar:
    end = cutoff + timedelta(days=18)
    if market != "india":
        return training_calendar(market, cutoff, end)
    start = cutoff - timedelta(days=PRICE_WARMUP_CALENDAR_DAYS)
    known = load_nse_cash_holidays(end, refresh_missing_years=False)
    if cutoff <= known.end:
        calendar = _india_calendar(start, min(end, known.end), known)
        anchor = calendar.anchor(cutoff)
        remaining = len(calendar.sessions) - calendar.sessions.searchsorted(
            pd.Timestamp(anchor), "right"
        )
        if remaining >= FORECAST_SESSIONS:
            return calendar
    return load_patchtst_calendar(market, start, end)

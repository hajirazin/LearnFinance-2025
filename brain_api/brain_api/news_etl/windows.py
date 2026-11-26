"""Coverage windows for Monday signals and PatchTST session-close features."""

from collections.abc import Sequence
from datetime import datetime
from typing import Literal

import pandas as pd

from brain_api.core.weekly_decision import canonical_monday_windows_contained_in
from brain_api.news.models import NewsWindow
from brain_api.news.patchtst import load_session_sentiment, session_news_window
from brain_api.news.store import NewsStore

NewsPurpose = Literal["monday", "patchtst"]


def requested_windows(
    start: datetime, end: datetime, purpose: NewsPurpose
) -> list[NewsWindow]:
    if start.tzinfo is None or end.tzinfo is None or start >= end:
        raise ValueError("News range must have ordered timezone-aware bounds")
    if purpose == "monday":
        return [
            NewsWindow(lo, hi)
            for lo, hi in canonical_monday_windows_contained_in(start, end)
        ]
    from brain_api.core.patchtst.calendar import load_patchtst_calendar

    calendar = load_patchtst_calendar("us", start.date(), end.date())
    return [
        window
        for close in calendar.closes
        if start < close <= end
        and (window := session_news_window(close)).start_exclusive >= start
    ]


def missing_session_windows(
    store: NewsStore, symbols: Sequence[str], windows: Sequence[NewsWindow]
) -> list[tuple[str, NewsWindow]]:
    if not windows or not symbols:
        return []
    # Use the same reconstruction and point-in-time rules as the forecaster.
    closes = pd.Series(
        [window.end_inclusive for window in windows], index=pd.RangeIndex(len(windows))
    )
    sentiments = load_session_sentiment(store, symbols, closes)
    return [
        (symbol, windows[index])
        for index in range(len(windows))
        for symbol in sentiments.columns[sentiments.iloc[index].isna()]
    ]

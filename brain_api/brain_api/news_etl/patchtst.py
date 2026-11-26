"""Prepare US completed-session sentiment before the weekly trading window."""

from datetime import UTC, date, datetime, timedelta

from brain_api.core.patchtst.calendar import load_patchtst_calendar
from brain_api.news.patchtst import load_session_sentiment, session_news_window
from brain_api.news.service import NewsService
from brain_api.news.store import NewsStore
from brain_api.news.sync import sync_news


def refresh_patchtst_news(store: NewsStore, *, session_date: date) -> str | None:
    from brain_api.universe.halal import get_halal_universe
    from brain_api.universe.halal_new import get_halal_new_universe

    calendar = load_patchtst_calendar(
        "us", session_date - timedelta(days=9), session_date
    )
    if session_date not in calendar.sessions.date:
        return None
    sync_news(store)
    broad = {item["symbol"] for item in get_halal_new_universe()["stocks"]}
    legacy = {item["symbol"] for item in get_halal_universe()["stocks"]}
    symbols = sorted(broad | legacy)
    if not symbols:
        raise ValueError("PatchTST news refresh resolved no symbols")
    closed = calendar.closes[calendar.closes <= datetime.now(UTC)]
    if closed.empty or closed.index[-1].date() != session_date:
        raise ValueError(f"US session {session_date} has not closed")
    service = NewsService(store)
    prepared = load_session_sentiment(store, symbols, closed)
    for session, close in closed.items():
        pending = prepared.columns[prepared.loc[session].isna()].tolist()
        window = session_news_window(close)
        for offset in range(0, len(pending), 40):
            service.materialize(pending[offset : offset + 40], window)
    commit = sync_news(store, publish=True)
    latest = load_session_sentiment(store, symbols, closed.iloc[-1:]).iloc[0].dropna()
    if len(set(latest.index) & broad) < 15 or len(set(latest.index) & legacy) < 10:
        raise ValueError(
            f"Insufficient compatible US PatchTST news coverage at {session_date}"
        )
    return commit

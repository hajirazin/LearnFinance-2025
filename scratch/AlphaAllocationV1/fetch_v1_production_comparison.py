"""Fetch read-only Alpha-HRP evidence and adjusted benchmark prices."""

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd
import requests
import yfinance as yf
from build_dataset import REPO, ROSTER, load_roster
from compare_patchtst import write_json
from dotenv import dotenv_values
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

OUTPUT = Path(__file__).resolve().parent / "data/v1_production_comparison"
START = pd.Timestamp("2026-08-21")
NY = ZoneInfo("America/New_York")
PAPER = "https://paper-api.alpaca.markets"


def fetch_broker(end, *, start=START, output=OUTPUT / "alpaca_evidence.json"):
    env = dotenv_values(REPO / "brain_api/.env")
    host = (env.get("ALPACA_HRP_URL") or PAPER).rstrip("/")
    if host != PAPER:
        raise ValueError("This comparison expects the dedicated HRP paper account")
    headers = {
        "APCA-API-KEY-ID": env["ALPACA_HRP_KEY"],
        "APCA-API-SECRET-KEY": env["ALPACA_HRP_SECRET"],
    }
    start_time = datetime.combine(start.date(), time.min, NY).isoformat()
    end_time = datetime.combine(end.date(), time.max, NY).isoformat()
    with requests.Session() as session:
        session.headers.update(headers)
        session.mount(
            PAPER,
            HTTPAdapter(
                max_retries=Retry(
                    total=3,
                    backoff_factor=1,
                    status_forcelist=[429, 500, 502, 503, 504],
                )
            ),
        )

        def get(path, params):
            response = session.get(PAPER + path, params=params, timeout=30)
            response.raise_for_status()
            return response.json()

        history = get(
            "/v2/account/portfolio/history",
            {
                "start": start_time,
                "end": end_time,
                "timeframe": "1D",
                "pnl_reset": "no_reset",
            },
        )
        if not history.get("equity"):
            raise ValueError("Empty Alpha-HRP equity history")
        latest_daily = datetime.fromtimestamp(history["timestamp"][-1], NY).date()
        latest_intraday = None
        if latest_daily < end.date():
            latest_intraday = get(
                "/v2/account/portfolio/history",
                {
                    "start": datetime.combine(end.date(), time.min, NY).isoformat(),
                    "end": datetime.now(NY).isoformat(),
                    "timeframe": "15Min",
                    "intraday_reporting": "market_hours",
                    "pnl_reset": "no_reset",
                },
            )
        activities, token = [], None
        while True:
            params = {
                "after": start_time,
                "until": end_time,
                "page_size": 100,
                "direction": "asc",
            }
            if token is not None:
                params["page_token"] = token
            page = get("/v2/account/activities", params)
            activities.extend(page)
            if len(page) < 100:
                break
            if page[-1]["id"] == token:
                raise ValueError("Alpaca activity pagination repeated a token")
            token = page[-1]["id"]
    write_json(
        output,
        {
            "fetched_at_utc": datetime.now(UTC).isoformat(),
            "start": start.date().isoformat(),
            "end": end.date().isoformat(),
            "source": "Alpaca Alpha-HRP paper account",
            "history": history,
            "latest_intraday": latest_intraday,
            "activities": activities,
        },
    )
    print(
        f"Alpaca returned {len(history['equity'])} equity observations and {len(activities)} activities",
        flush=True,
    )


def fetch_prices(end):
    roster = load_roster(ROSTER)
    symbols = sorted(set(roster["symbols"]) | {"SPY"})
    yf.set_tz_cache_location(str(OUTPUT / "yfinance_cache"))
    frames = []
    for i in range(0, len(symbols), 40):
        batch = symbols[i : i + 40]
        bars = yf.download(
            batch,
            start=START.date().isoformat(),
            end=(end.date() + timedelta(days=1)).isoformat(),
            auto_adjust=False,
            progress=False,
            threads=6,
            group_by="column",
        )
        frames.append(bars["Adj Close"].reindex(columns=batch))
        print(
            f"Requested prices for {i + len(batch)}/{len(symbols)} symbols", flush=True
        )
    closes = pd.concat(frames, axis=1).sort_index()
    closes.index = closes.index.tz_localize(None)
    closes.rename_axis("date").to_csv(OUTPUT / "adjusted_close.csv")
    write_json(
        OUTPUT / "price_manifest.json",
        {
            "fetched_at_utc": datetime.now(UTC).isoformat(),
            "source": "Yahoo Finance dividend- and split-adjusted daily closes",
            "roster": roster,
            "start": START.date().isoformat(),
            "end": end.date().isoformat(),
            "no_prices": closes.columns[closes.isna().all()].tolist(),
        },
    )


def main():
    now = pd.Timestamp.now(tz="UTC")
    calendar = xcals.get_calendar("XNYS")
    day = pd.Timestamp(now.tz_convert(NY).date())
    sessions = calendar.sessions_in_range(day - pd.Timedelta(days=14), day)
    end = next(
        session
        for session in reversed(sessions)
        if calendar.session_close(session) < now
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = (pool.submit(fetch_broker, end), pool.submit(fetch_prices, end))
        for job in jobs:
            job.result()


if __name__ == "__main__":
    main()

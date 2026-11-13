"""Cache independent SIP closes for the Friday price audit."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
from audit_data import DATA
from build_dataset import REPO

EVIDENCE = DATA / "friday" / "alpaca_sip_evidence.json"
URL = "https://data.alpaca.markets/v2/stocks/bars"


def get_bars(headers, symbols, start, end, adjustment):
    query = {
        "symbols": ",".join(symbols),
        "timeframe": "1Day",
        "start": start.date().isoformat(),
        "end": (end + pd.Timedelta(days=1)).date().isoformat(),
        "feed": "sip",
        "adjustment": adjustment,
        "asof": "-",
        "limit": 10000,
    }
    closes = {}
    while True:
        request = Request(f"{URL}?{urlencode(query)}", headers=headers)
        for attempt in range(5):
            try:
                with urlopen(request, timeout=30) as response:
                    data = json.load(response)
                break
            except HTTPError as exc:
                if attempt == 4 or (exc.code != 429 and exc.code < 500):
                    raise
                time.sleep(2**attempt)
        for symbol, bars in (data.get("bars") or {}).items():
            for bar in bars:
                closes[(symbol, bar["t"][:10])] = [bar["c"], bar["v"]]
        token = data.get("next_page_token")
        if not token:
            return closes
        query["page_token"] = token


def fetch_evidence(targets):
    from brain_api.core.config import get_alpaca_api_key, get_alpaca_api_secret
    from dotenv import load_dotenv

    load_dotenv(REPO / "brain_api/.env")
    headers = {
        "APCA-API-KEY-ID": get_alpaca_api_key(),
        "APCA-API-SECRET-KEY": get_alpaca_api_secret(),
    }
    if not all(headers.values()):
        raise ValueError("Set the Alpaca credentials in brain_api/.env")

    def fetch_month(group):
        dates = group["date"]
        wanted = set(zip(group["symbol"], dates.dt.strftime("%Y-%m-%d"), strict=True))
        closes = get_bars(
            headers, sorted(group["symbol"].unique()), dates.min(), dates.max(), "all"
        )
        return [
            [symbol, day, *closes[(symbol, day)]]
            for symbol, day in sorted(wanted & closes.keys())
        ]

    groups = [group for _, group in targets.groupby(targets["date"].dt.to_period("M"))]
    with ThreadPoolExecutor(max_workers=4) as executor:
        adjusted = [row for month in executor.map(fetch_month, groups) for row in month]
    raw = get_bars(
        headers,
        ["DHR", "FTV"],
        pd.Timestamp("2016-07-01"),
        pd.Timestamp("2016-07-08"),
        "raw",
    )
    payload = {
        "source": URL,
        "feed": "sip",
        "asof": "-",
        "fetched_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "requested_bars": len(targets),
        "adjusted_closes": adjusted,
        "fortive_raw_closes": [
            [symbol, day, *bar] for (symbol, day), bar in sorted(raw.items())
        ],
    }
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    staged = EVIDENCE.with_suffix(".tmp")
    staged.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    staged.replace(EVIDENCE)

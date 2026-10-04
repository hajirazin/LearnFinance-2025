"""Read-only full order archive and public traded-symbol price evidence."""

import json
import pickle
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

from brain_api.core.alpaca_client import get_alpaca_client
from brain_api.core.prices import load_close_prices_yfinance

OUT = Path(__file__).parent
ROOT = OUT.resolve().parents[1]
load_dotenv(ROOT / "brain_api/.env")
archive = {"fetched_at": datetime.now(timezone.utc).isoformat(), "accounts": {}}
symbols = {"SPY", "QQQ", "SPUS", "HLAL"}
with httpx.Client(timeout=60) as http:
    for account in ["hrp", "dhrp", "sac"]:
        if "--prices-only" in sys.argv:
            previous = json.loads((OUT / "full_order_archive.json").read_text())
            symbols.update(o["symbol"] for o in previous["accounts"][account]["orders"])
            continue
        client = get_alpaca_client(account)
        orders = []
        counts = {}
        for month in range(1, 10):
            start = f"2026-{month:02d}-01T00:00:00Z"
            end = f"2026-{month + 1:02d}-01T00:00:00Z"
            response = http.get(
                client.base_url + "/v2/orders",
                headers=client._headers(),
                params={
                    "status": "all",
                    "after": start,
                    "until": end,
                    "limit": 500,
                    "direction": "asc",
                },
            )
            response.raise_for_status()
            rows = response.json()
            if len(rows) >= 500:
                raise RuntimeError(f"{account} {month} order page may be truncated")
            orders.extend(rows)
            counts[str(month)] = len(rows)
        orders = list({o["id"]: o for o in orders}.values())
        symbols.update(o["symbol"] for o in orders)
        archive["accounts"][account] = {
            "host": client.base_url,
            "month_counts": counts,
            "orders": [
                {
                    k: o.get(k)
                    for k in [
                        "symbol",
                        "side",
                        "qty",
                        "filled_qty",
                        "filled_avg_price",
                        "status",
                        "created_at",
                        "filled_at",
                        "client_order_id",
                        "type",
                        "limit_price",
                        "notional",
                    ]
                }
                for o in orders
            ],
        }
        print(account, "downloaded orders", len(orders), flush=True)
if "--prices-only" not in sys.argv:
    (OUT / "full_order_archive.json").write_text(json.dumps(archive, indent=2))
# Include the frozen current universe for same-week cross-sectional comparisons.
universe = json.loads(
    (ROOT / "brain_api/data/cache/universe/halal_new_2026-09.json").read_text()
)
symbols.update(s["symbol"] for s in universe["stocks"])
prices = load_close_prices_yfinance(
    sorted(symbols), date(2025, 1, 1), date(2026, 10, 2)
)
(OUT / "full_price_evidence.pkl").write_bytes(pickle.dumps(prices))
(OUT / "full_price_manifest.json").write_text(
    json.dumps(
        {
            "source": "Yahoo auto-adjusted OHLCV, close validity only",
            "start": "2025-01-01",
            "end_inclusive": "2026-10-02",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "requested": sorted(symbols),
            "missing": sorted(symbols - set(prices)),
        },
        indent=2,
    )
)
print("Public prices", len(prices), "of", len(symbols), flush=True)

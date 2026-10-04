"""Read-only Alpaca account/price evidence. No submission methods are called."""

import json
from datetime import datetime, timezone
from pathlib import Path
import httpx
from dotenv import load_dotenv
from brain_api.core.alpaca_client import get_alpaca_client

OUT = Path(__file__).parent
ROOT = OUT.resolve().parents[1]
load_dotenv(ROOT / "brain_api/.env")
out = {"fetched_at": datetime.now(timezone.utc).isoformat(), "accounts": {}}
for name in ["hrp", "dhrp", "sac"]:
    client = get_alpaca_client(name)
    with httpx.Client(timeout=60) as http:

        def get(path, params=None):
            response = http.get(
                client.base_url + path, headers=client._headers(), params=params
            )
            response.raise_for_status()
            return response.json()

        account = get("/v2/account")
        portfolio = get(
            "/v2/account/portfolio/history",
            {
                "date_start": "2026-01-01",
                "date_end": "2026-10-03",
                "timeframe": "1D",
                "extended_hours": "false",
                "pnl_reset": "no_reset",
            },
        )
        orders = get(
            "/v2/orders",
            {
                "status": "all",
                "after": "2026-07-01T00:00:00Z",
                "until": "2026-10-03T00:00:00Z",
                "limit": 500,
                "direction": "asc",
            },
        )
        positions = get("/v2/positions")
        activities = get(
            "/v2/account/activities",
            {
                "after": "2026-01-01",
                "until": "2026-10-03",
                "page_size": 100,
                "direction": "asc",
                "activity_types": "CSD,CSW",
            },
        )
        recent_activities = get(
            "/v2/account/activities",
            {
                "after": "2026-09-23",
                "until": "2026-10-03",
                "page_size": 100,
                "direction": "asc",
            },
        )
        out["accounts"][name] = {
            "host": client.base_url,
            "account": {
                k: account.get(k)
                for k in [
                    "equity",
                    "cash",
                    "last_equity",
                    "buying_power",
                    "portfolio_value",
                ]
            },
            "portfolio_history": portfolio,
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
                    ]
                }
                for o in orders
            ],
            "positions": [
                {
                    k: p.get(k)
                    for k in [
                        "symbol",
                        "qty",
                        "avg_entry_price",
                        "current_price",
                        "market_value",
                        "unrealized_pl",
                    ]
                }
                for p in positions
            ],
            "cash_flows": activities,
        }
        print(
            name,
            "equity",
            account["equity"],
            "daily points",
            len(portfolio.get("equity", [])),
            "orders",
            len(orders),
        )
        out["accounts"][name]["recent_activities"] = recent_activities
(OUT / "broker_history.json").write_text(json.dumps(out, indent=2))

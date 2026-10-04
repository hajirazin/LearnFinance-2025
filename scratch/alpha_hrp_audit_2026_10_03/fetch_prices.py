"""Read public price evidence for the October Alpha-HRP audit."""

import json
import pickle
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from brain_api.core.prices import load_prices_yfinance

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
if "--market-only" in sys.argv:
    market_symbols = [
        "SPY",
        "QQQ",
        "RSP",
        "IWM",
        "^VIX",
        "^TNX",
        "TLT",
        "XLE",
        "XLK",
        "XLV",
        "XLF",
        "XLI",
        "XLY",
        "XLP",
        "XLU",
        "XLB",
        "XLRE",
        "XLC",
        "SMH",
        "GLD",
        "SPUS",
        "HLAL",
        "SPTE",
        "SPWO",
        "UMMA",
    ]
    market = load_prices_yfinance(market_symbols, date(2025, 1, 1), date(2026, 10, 2))
    with (OUT / "market_prices.pkl").open("wb") as handle:
        pickle.dump(market, handle)
    (OUT / "market_manifest.json").write_text(
        json.dumps(
            {
                "source": "Yahoo auto-adjusted daily OHLCV; VIX/TNX are index levels",
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "start": "2025-01-01",
                "end_inclusive": "2026-10-02",
                "requested": market_symbols,
                "missing": sorted(set(market_symbols) - set(market)),
            },
            indent=2,
        )
    )
    print("Saved market conditions:", len(market), "/", len(market_symbols))
    raise SystemExit(0)
universe_path = ROOT / "brain_api/data/cache/universe/halal_new_2026-09.json"
universe = json.loads(universe_path.read_text())
symbols = sorted({s["symbol"] for s in universe["stocks"]} | {"SPY", "QQQ"})
prices = load_prices_yfinance(symbols, date(2026, 4, 1), date(2026, 10, 2))
with (OUT / "prices.pkl").open("wb") as handle:
    pickle.dump(prices, handle)
(OUT / "price_manifest.json").write_text(
    json.dumps(
        {
            "source": "Yahoo via production load_prices_yfinance; auto_adjust=True",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "start": "2026-04-01",
            "end_inclusive": "2026-10-02",
            "requested": symbols,
            "missing": sorted(set(symbols) - set(prices)),
            "universe_source": str(universe_path),
        },
        indent=2,
    )
)
print("Saved public price evidence:", len(prices), "/", len(symbols))

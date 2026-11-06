"""Explicit same-provider retry of the failed SMCAY cache transaction."""

import json
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
yf.set_tz_cache_location(str(ROOT / "yfinance_cache"))
data = yf.download(
    "SMCAY",
    start="2019-01-01",
    end="2026-09-15",
    auto_adjust=True,
    threads=False,
    progress=False,
    timeout=20,
)
if data.empty:
    raise RuntimeError(
        "SMCAY provider retry failed; cannot silently omit a cache failure"
    )
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)
frame = data.reindex(columns=["Open", "High", "Low", "Close", "Volume"])
frame.columns = ["open", "high", "low", "close", "volume"]
prices = pd.read_pickle(ROOT / "market_prices.pkl")
prices["SMCAY"] = frame.dropna()
pd.to_pickle(prices, ROOT / "market_prices.pkl")
manifest = json.loads((ROOT / "market_download.json").read_text())
manifest["rows"]["SMCAY"] = len(prices["SMCAY"])
manifest["empty_symbols"] = [s for s, f in prices.items() if f.empty]
manifest["explicit_retries"] = {
    "SMCAY": {"reason": "yfinance cache database locked", "rows": len(prices["SMCAY"])}
}
(ROOT / "market_download.json").write_text(json.dumps(manifest, indent=2))
print("SMCAY retry rows", len(prices["SMCAY"]), flush=True)

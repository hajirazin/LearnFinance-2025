"""Scrape full holdings from halal ETF provider websites and save as JSON.

Usage:
    .venv/bin/python scripts/scrape_halal_holdings.py          # use cached per-ETF JSONs if they exist
    .venv/bin/python scripts/scrape_halal_holdings.py --fresh   # force re-download all ETFs

Outputs JSON files to scripts/output/:
    - {etf}_holdings.json        (per-ETF: spus, spte, spwo, hlal, umma)
    - merged_all.json            (deduplicated, before any filter)
    - merged_tradable.json       (after Alpaca tradable + excluded filter)
    - filtered_out.json          (not tradable and excluded tickers)
    - alpaca_tradable_assets.json (Alpaca asset summary)
"""

import csv
import io
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / "brain_api" / ".env")

# Tickers to exclude even if tradable on Alpaca
# - GOOG: Alphabet Class C (duplicate of GOOGL Class A)
EXCLUDED_TICKERS = {"GOOG"}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# All halal ETFs to scrape, grouped by data source
SP_FUNDS_ETFS = ["spus", "spte", "spwo"]
WAHED_ETFS = ["hlal", "umma"]

# Wahed holdings as public Google Sheets (full daily holdings with tickers).
# Export as CSV via: https://docs.google.com/spreadsheets/d/{ID}/export?format=csv
WAHED_GSHEET_IDS = {
    "hlal": "1UC1Bk67bGuYsos_i8y_HQpNoHpVHAvqf71MbgrafJOQ",
    "umma": "1kACYezLTfiN5dWMrM02GL2uQWsYTj2nqVTejp6hJp2k",
}


# ---------------------------------------------------------------------------
# Alpaca
# ---------------------------------------------------------------------------


def fetch_alpaca_tradable_symbols() -> set[str]:
    """Fetch all tradable symbols from Alpaca's GET /v2/assets endpoint."""
    api_key = os.environ.get("ALPACA_API_KEY", "")
    api_secret = os.environ.get("ALPACA_API_SECRET", "")
    if not api_key or not api_secret:
        raise RuntimeError(
            "ALPACA_API_KEY and ALPACA_API_SECRET must be set in brain_api/.env"
        )

    resp = httpx.get(
        "https://paper-api.alpaca.markets/v2/assets",
        headers={
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret,
        },
        params={"status": "active", "asset_class": "us_equity"},
        timeout=60,
    )
    resp.raise_for_status()
    all_assets = resp.json()

    tradable = [a for a in all_assets if a.get("tradable")]
    tradable_symbols = {a["symbol"] for a in tradable}

    print(f"  Alpaca: {len(all_assets)} active US equities, {len(tradable)} tradable")
    return tradable_symbols


def is_tradable(symbol: str, alpaca_symbols: set[str]) -> bool:
    """Check if a ticker is tradable on Alpaca."""
    return symbol in alpaca_symbols


# ---------------------------------------------------------------------------
# Scrapers
# ---------------------------------------------------------------------------


def scrape_sp_funds(etf_slug: str) -> list[dict]:
    """Scrape full holdings from sp-funds.com."""
    resp = requests.get(
        f"https://www.sp-funds.com/{etf_slug}/", timeout=30, headers=HEADERS
    )
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    holdings = []
    for table in soup.find_all("table"):
        header_cells = [th.get_text(strip=True) for th in table.find_all("th")]
        if "StockTicker" not in header_cells:
            continue

        ti = header_cells.index("StockTicker")
        ni = (
            header_cells.index("SecurityName")
            if "SecurityName" in header_cells
            else None
        )
        wi = header_cells.index("Weightings") if "Weightings" in header_cells else None

        for row in table.find_all("tr")[1:]:
            cells = row.find_all("td")
            if len(cells) <= ti:
                continue
            ticker = cells[ti].get_text(strip=True)
            name = cells[ni].get_text(strip=True) if ni and len(cells) > ni else ""
            weight_s = cells[wi].get_text(strip=True) if wi and len(cells) > wi else "0"
            try:
                weight = float(weight_s)
            except ValueError:
                weight = 0.0
            if ticker and ticker != "Cash&Other":
                holdings.append({"symbol": ticker, "name": name, "weight": weight})
        break

    return holdings


def scrape_wahed_gsheet(etf_slug: str) -> list[dict]:
    """Fetch full holdings from a Wahed ETF Google Sheet (exported as CSV).

    The sheets have columns: Date, Account, StockTicker, CUSIP, SecurityName,
    Shares, Price, MarketValue, Weightings, ...
    Tickers for foreign stocks use exchange suffixes (e.g. "005930 KS", "ASML NA").
    US-listed tickers are plain (e.g. "AAPL", "TSM", "INFY").
    """
    sheet_id = WAHED_GSHEET_IDS[etf_slug]
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    reader = csv.DictReader(io.StringIO(resp.text))
    holdings: list[dict] = []
    seen_tickers: set[str] = set()
    for row in reader:
        ticker = row.get("StockTicker", "").strip()
        if not ticker or ticker == "Cash&Other":
            continue

        # Use the raw ticker as-is (the Alpaca filter will handle non-US ones)
        # But strip exchange suffixes for foreign tickers so they can potentially
        # match Alpaca ADR symbols (e.g. "ASML NA" won't match, but "INFY" will).
        symbol = ticker.split()[0] if " " in ticker else ticker

        name = row.get("SecurityName", "").strip()
        weight_s = row.get("Weightings", "0").strip().rstrip("%")
        try:
            weight = float(weight_s)
        except ValueError:
            weight = 0.0

        # Deduplicate (sheets may have multiple date rows)
        if symbol in seen_tickers:
            continue
        seen_tickers.add(symbol)

        holdings.append({"symbol": symbol, "name": name, "weight": weight})

    return holdings


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def load_cached(etf_slug: str) -> list[dict] | None:
    """Load holdings from a previously saved JSON file if it exists."""
    path = OUTPUT_DIR / f"{etf_slug}_holdings.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("holdings")


def fetch_etf(etf_slug: str, force: bool = False) -> list[dict]:
    """Fetch holdings for an ETF, using cache unless force=True.

    Returns:
        List of holdings dicts with symbol, name, weight.
    """
    if not force:
        cached = load_cached(etf_slug)
        if cached is not None:
            print(f"  {etf_slug.upper()}: {len(cached)} holdings (cached)")
            return cached

    if etf_slug in SP_FUNDS_ETFS:
        print(f"  {etf_slug.upper()}: scraping sp-funds.com...")
        holdings = scrape_sp_funds(etf_slug)
    elif etf_slug in WAHED_ETFS:
        print(f"  {etf_slug.upper()}: fetching from Wahed Google Sheet...")
        holdings = scrape_wahed_gsheet(etf_slug)
    else:
        raise ValueError(f"Unknown ETF: {etf_slug}")

    print(f"  {etf_slug.upper()}: {len(holdings)} holdings (downloaded)")
    return holdings


# ---------------------------------------------------------------------------
# Merge / Save
# ---------------------------------------------------------------------------


def merge_and_dedup(etf_holdings: dict[str, list[dict]]) -> list[dict]:
    """Merge holdings from multiple ETFs, dedup by symbol, track sources."""
    all_h: dict[str, dict] = {}
    for etf_name, holdings in etf_holdings.items():
        for h in holdings:
            sym = h["symbol"]
            if sym not in all_h:
                all_h[sym] = {
                    "symbol": sym,
                    "name": h["name"],
                    "max_weight": h["weight"],
                    "sources": [etf_name],
                }
            else:
                existing = all_h[sym]
                existing["max_weight"] = max(existing["max_weight"], h["weight"])
                if etf_name not in existing["sources"]:
                    existing["sources"].append(etf_name)

    return sorted(all_h.values(), key=lambda x: x["max_weight"], reverse=True)


def save_json(data: object, filename: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    force = "--fresh" in sys.argv
    ts = datetime.now(UTC).isoformat()

    all_etfs = SP_FUNDS_ETFS + WAHED_ETFS

    # 1. Fetch Alpaca tradable symbols (always fresh -- fast & no scrape risk)
    print("Fetching Alpaca tradable assets...")
    alpaca_symbols = fetch_alpaca_tradable_symbols()

    # 2. Fetch each ETF (cached or fresh)
    print(f"\nFetching {len(all_etfs)} ETFs (use --fresh to force re-download)...")
    etf_data: dict[str, list[dict]] = {}
    for slug in all_etfs:
        holdings = fetch_etf(slug, force=force)
        etf_data[slug.upper()] = holdings

    # 3. Save per-ETF JSONs (only if freshly downloaded)
    for slug in all_etfs:
        name_upper = slug.upper()
        data = etf_data[name_upper]
        p = save_json(
            {"etf": name_upper, "fetched_at": ts, "count": len(data), "holdings": data},
            f"{slug}_holdings.json",
        )
        print(f"  Saved {p}")

    # 4. Merge & dedup
    merged = merge_and_dedup(etf_data)
    p = save_json(
        {"fetched_at": ts, "total_unique": len(merged), "holdings": merged},
        "merged_all.json",
    )
    print(f"\nMerged (all): {len(merged)} unique tickers -> {p}")

    # 5. Filter: Alpaca tradable + excluded
    tradable_stocks = []
    not_tradable = []
    excluded = []
    for h in merged:
        sym = h["symbol"]
        if not is_tradable(sym, alpaca_symbols):
            not_tradable.append(h)
        elif sym in EXCLUDED_TICKERS:
            excluded.append(h)
        else:
            tradable_stocks.append(h)

    p = save_json(
        {
            "fetched_at": ts,
            "total_tradable_stocks": len(tradable_stocks),
            "holdings": tradable_stocks,
        },
        "merged_tradable.json",
    )
    print(f"\nTradable on Alpaca: {len(tradable_stocks)} stocks -> {p}")

    p = save_json(
        {
            "fetched_at": ts,
            "not_tradable": not_tradable,
            "excluded": excluded,
            "not_tradable_count": len(not_tradable),
            "excluded_count": len(excluded),
        },
        "filtered_out.json",
    )
    print(
        f"Filtered out: {len(not_tradable)} not tradable + {len(excluded)} excluded -> {p}"
    )

    # 6. Save Alpaca summary
    p = save_json(
        {
            "fetched_at": ts,
            "total_tradable_on_alpaca": len(alpaca_symbols),
            "tradable_symbols": sorted(alpaca_symbols),
        },
        "alpaca_tradable_assets.json",
    )
    print(f"Alpaca assets -> {p}")


if __name__ == "__main__":
    main()

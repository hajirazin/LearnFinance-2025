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
    - junk_filtered.json         (after ROE > 0, Price > SMA200, Beta < 2)
    - factor_scored.json         (ranked by 0.4*Momentum + 0.3*Quality + 0.3*Value)
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
# Stock metrics (yfinance)
# ---------------------------------------------------------------------------


def fetch_stock_metrics(symbols: list[str]) -> dict[str, dict]:
    """Fetch fundamental metrics for each symbol using yfinance .info.

    Uses ThreadPoolExecutor for parallelism.  Also does a single batch
    yf.download() call for 6-month price history (momentum calculation).

    Returns:
        {symbol: {roe, price, sma200, beta, gross_margin, roic,
                  earnings_yield, six_month_return, ...}}
    """
    import concurrent.futures

    import yfinance as yf

    # -- Batch download 6-month prices for momentum ---
    print("  Downloading 6-month price history (batch)...")
    hist = yf.download(symbols, period="6mo", progress=False, threads=True)
    six_month_returns: dict[str, float | None] = {}
    if "Close" in hist.columns:
        close = hist["Close"]
        if isinstance(close, dict):
            close_df = close
        else:
            close_df = close
        for sym in symbols:
            try:
                series = close_df[sym].dropna()
                if len(series) >= 2:
                    ret = (series.iloc[-1] / series.iloc[0]) - 1
                    six_month_returns[sym] = float(ret)
                else:
                    six_month_returns[sym] = None
            except (KeyError, TypeError):
                six_month_returns[sym] = None

    # -- Individual .info calls via thread pool ---
    print(f"  Fetching .info for {len(symbols)} stocks (threaded)...")

    def _fetch_one(sym: str) -> tuple[str, dict]:
        try:
            info = yf.Ticker(sym).info
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            op_margins = info.get("operatingMargins")
            total_rev = info.get("totalRevenue")
            ev = info.get("enterpriseValue")
            book = info.get("bookValue")
            shares = info.get("sharesOutstanding")
            total_debt = info.get("totalDebt") or 0
            total_cash = info.get("totalCash") or 0

            # ROIC ≈ NOPAT / Invested Capital
            # NOPAT ≈ Operating Income × 0.75 (assumed 25% tax)
            # Invested Capital ≈ Total Equity + Total Debt - Cash
            roic = None
            if op_margins and total_rev and book and shares:
                operating_income = op_margins * total_rev
                invested_capital = (book * shares) + total_debt - total_cash
                if invested_capital > 0:
                    roic = (operating_income * 0.75) / invested_capital

            # Earnings Yield = EBIT / EV
            earnings_yield = None
            if op_margins and total_rev and ev and ev > 0:
                ebit = op_margins * total_rev
                earnings_yield = ebit / ev

            return sym, {
                "roe": info.get("returnOnEquity"),
                "price": price,
                "sma200": info.get("twoHundredDayAverage"),
                "beta": info.get("beta"),
                "gross_margin": info.get("grossMargins"),
                "roic": roic,
                "earnings_yield": earnings_yield,
                "six_month_return": six_month_returns.get(sym),
            }
        except Exception:
            return sym, {}

    metrics: dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(_fetch_one, sym): sym for sym in symbols}
        done = 0
        for future in concurrent.futures.as_completed(futures):
            sym, data = future.result()
            metrics[sym] = data
            done += 1
            if done % 50 == 0:
                print(f"    {done}/{len(symbols)} done...")

    print(f"  Fetched metrics for {len(metrics)} stocks")
    return metrics


# ---------------------------------------------------------------------------
# Step 1: Junk Filter
# ---------------------------------------------------------------------------


def apply_junk_filter(
    holdings: list[dict], metrics: dict[str, dict]
) -> tuple[list[dict], list[dict]]:
    """Remove stocks that fail basic quality checks.

    Rules:
        1. ROE > 0  (company must be making money)
        2. Price > 200-day SMA  (don't catch falling knives)
        3. Beta < 2.0  (exclude wildly unstable stocks)

    Returns:
        (passed, failed) - both lists of holdings dicts with metrics attached.
    """
    passed: list[dict] = []
    failed: list[dict] = []

    for h in holdings:
        sym = h["symbol"]
        m = metrics.get(sym, {})
        roe = m.get("roe")
        price = m.get("price")
        sma200 = m.get("sma200")
        beta = m.get("beta")

        reasons: list[str] = []
        if roe is None or roe <= 0:
            reasons.append(f"ROE={roe}")
        if price is None or sma200 is None or price <= sma200:
            reasons.append(f"Price={price} <= SMA200={sma200}")
        if beta is None or beta >= 2.0:
            reasons.append(f"Beta={beta}")

        entry = {**h, "metrics": m}
        if reasons:
            entry["fail_reasons"] = reasons
            failed.append(entry)
        else:
            passed.append(entry)

    return passed, failed


# ---------------------------------------------------------------------------
# Step 2: Factor Score
# ---------------------------------------------------------------------------


def _percentile_rank(values: list[float | None]) -> list[float | None]:
    """Compute percentile rank (0-1) for each value. None stays None."""
    valid = [(i, v) for i, v in enumerate(values) if v is not None]
    if not valid:
        return [None] * len(values)
    sorted_vals = sorted(v for _, v in valid)
    n = len(sorted_vals)
    ranks = [None] * len(values)
    for i, v in valid:
        pos = sorted_vals.index(v)
        ranks[i] = pos / max(n - 1, 1)
    return ranks


def compute_factor_scores(holdings: list[dict]) -> list[dict]:
    """Rank stocks by composite factor score.

    Score = 0.4 × Momentum + 0.3 × Quality + 0.3 × Value

    Momentum : 6-month return (percentile rank)
    Quality  : ROIC, falling back to Gross Margin (percentile rank)
    Value    : Earnings Yield = EBIT/EV (percentile rank)

    Returns:
        Holdings sorted by score descending, with score & components attached.
    """
    # Extract raw values
    momentum_raw = [h.get("metrics", {}).get("six_month_return") for h in holdings]
    quality_raw = []
    for h in holdings:
        m = h.get("metrics", {})
        quality_raw.append(
            m.get("roic") if m.get("roic") is not None else m.get("gross_margin")
        )
    value_raw = [h.get("metrics", {}).get("earnings_yield") for h in holdings]

    # Percentile ranks (0 = worst, 1 = best)
    momentum_pct = _percentile_rank(momentum_raw)
    quality_pct = _percentile_rank(quality_raw)
    value_pct = _percentile_rank(value_raw)

    scored: list[dict] = []
    for i, h in enumerate(holdings):
        mp = momentum_pct[i]
        qp = quality_pct[i]
        vp = value_pct[i]

        # If any component is missing, score = None (ranked last)
        if mp is not None and qp is not None and vp is not None:
            score = round(0.4 * mp + 0.3 * qp + 0.3 * vp, 6)
        else:
            score = None

        scored.append(
            {
                **h,
                "factor_score": score,
                "factor_components": {
                    "momentum_6m_return": momentum_raw[i],
                    "momentum_pct": mp,
                    "quality_raw": quality_raw[i],
                    "quality_pct": qp,
                    "value_earnings_yield": value_raw[i],
                    "value_pct": vp,
                },
            }
        )

    scored.sort(
        key=lambda x: x.get("factor_score")
        if x.get("factor_score") is not None
        else -1,
        reverse=True,
    )
    return scored


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

    # ------------------------------------------------------------------
    # 7. Junk Filter + Factor Score (uses yfinance -- always fresh)
    # ------------------------------------------------------------------
    tradable_syms = [h["symbol"] for h in tradable_stocks]

    print(f"\nFetching yfinance metrics for {len(tradable_syms)} tradable stocks...")
    stock_metrics = fetch_stock_metrics(tradable_syms)

    # Step 1: Junk Filter
    junk_passed, junk_failed = apply_junk_filter(tradable_stocks, stock_metrics)

    p = save_json(
        {
            "fetched_at": ts,
            "description": "Tradable halal stocks passing junk filter: ROE > 0, Price > SMA200, Beta < 2",
            "total_passed": len(junk_passed),
            "total_failed": len(junk_failed),
            "passed": junk_passed,
            "failed": junk_failed,
        },
        "junk_filtered.json",
    )
    print(f"\nJunk filter: {len(junk_passed)} passed, {len(junk_failed)} failed -> {p}")

    # Step 2: Factor Score
    scored = compute_factor_scores(junk_passed)

    p = save_json(
        {
            "fetched_at": ts,
            "description": "Score = 0.4*Momentum(6m return) + 0.3*Quality(ROIC/GrossMargin) + 0.3*Value(EBIT/EV)",
            "total_scored": len(scored),
            "stocks": scored,
        },
        "factor_scored.json",
    )
    print(f"Factor scored: {len(scored)} stocks ranked -> {p}")

    # Print top 10
    print("\nTop 10 by factor score:")
    for i, s in enumerate(scored[:10]):
        sym = s["symbol"]
        score = s.get("factor_score", "N/A")
        fc = s.get("factor_components", {})
        mom = fc.get("momentum_6m_return")
        mom_s = f"{mom:+.1%}" if mom is not None else "N/A"
        print(f"  {i + 1:2d}. {sym:8s} score={score:.4f}  6m={mom_s}")


if __name__ == "__main__":
    main()

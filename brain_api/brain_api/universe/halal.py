"""Halal stock universe from ETF holdings."""

from datetime import UTC, datetime

import yfinance as yf

# Halal ETFs to source holdings from
HALAL_ETFS = ["SPUS", "HLAL", "SPTE"]


def _fetch_etf_holdings(ticker: str) -> list[dict]:
    """Fetch top holdings for a single ETF using yfinance.

    Args:
        ticker: ETF ticker symbol

    Returns:
        List of holdings with symbol, name, weight
    """
    try:
        etf = yf.Ticker(ticker)
        if not hasattr(etf, "funds_data") or etf.funds_data is None:
            return []

        top_holdings = etf.funds_data.top_holdings
        if top_holdings is None or top_holdings.empty:
            return []

        rows = []
        for symbol, row in top_holdings.iterrows():
            # yfinance returns weight as decimal (0.15 = 15%)
            weight_raw = row.get("Holding Percent", row.get("holdingPercent", 0))
            weight = float(weight_raw) * 100 if weight_raw else 0

            rows.append(
                {
                    "symbol": str(symbol),
                    "name": row.get("Name", row.get("holdingName", "")),
                    "weight": weight,
                }
            )
        return rows
    except Exception:
        return []


def get_halal_universe() -> dict:
    """Fetch and merge top holdings from halal ETFs.

    Returns:
        Dict with:
        - stocks: deduplicated list sorted by max_weight desc
        - etfs_used: list of ETF tickers queried
        - total_stocks: count of unique stocks
        - fetched_at: ISO timestamp
    """
    # Collect holdings from all ETFs
    all_holdings: dict[str, dict] = {}

    for etf_ticker in HALAL_ETFS:
        holdings = _fetch_etf_holdings(etf_ticker)
        for h in holdings:
            symbol = h["symbol"]
            if symbol not in all_holdings:
                all_holdings[symbol] = {
                    "symbol": symbol,
                    "name": h["name"],
                    "max_weight": h["weight"],
                    "sources": [etf_ticker],
                }
            else:
                # Update max weight and add source
                existing = all_holdings[symbol]
                existing["max_weight"] = max(existing["max_weight"], h["weight"])
                if etf_ticker not in existing["sources"]:
                    existing["sources"].append(etf_ticker)

    # Sort by max_weight descending
    stocks = sorted(all_holdings.values(), key=lambda x: x["max_weight"], reverse=True)

    return {
        "stocks": stocks,
        "etfs_used": HALAL_ETFS,
        "total_stocks": len(stocks),
        "fetched_at": datetime.now(UTC).isoformat(),
    }
